/*
Kernels for attention backward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt attention_backward.cu -o attention_backward

version 1 is a naive first version
OMP_NUM_THREADS=32 ./attention_backward 1

version 2 much ensures better load-balancing by having independent threads for each batch and attention head
OMP_NUM_THREADS=32 ./attention_backward 2

version 3 uses a full warp to calculate each result (instead of a thread), which enables coalesced memory access
OMP_NUM_THREADS=32 ./attention_backward 3

version 4 improves data reuse in registers by doing 8 values of t3 in one warp.
OMP_NUM_THREADS=32 ./attention_backward 4

version 5 reduces the amount of non-fp32 instructions needed by avoiding ifs
OMP_NUM_THREADS=32 ./attention_backward 5
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

/*
NOTE:
This version of attention_forward is modified to be consistent with the
attention_forward GPU kernel in the following way small but important way:
- preatt is only QUERY @ KEY, without the scale
- the scale instead moved and fused into the softmax
- the full preatt matrix is materialized, even the parts that get masked out
    - this doesn't actually change anything due to masking, but it lets us
      easily compare to the GPU version, which also does the full, dense sgemm
In this way we'll be able to make sure that preatt and att agree CPU vs GPU
*/
void attention_forward_cpu(float* out, float* preatt, float* att,
                            float* inp,
                            int B, int T, int C, int NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -FLT_MAX;
                for (int t2 = 0; t2 < T; t2++) { // used to be t2 <= t
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(scale * (preatt_bth[t2] - maxval));
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

// NOTE: Also contains the re-shuffling of the exact position of "scale"
// and when it is applied (after preatt, not "during" preatt)
// also, full matrices are materialized, even the parts that get masked out
void attention_backward_cpu(float* dinp, float* dpreatt, float* datt,
                            float* dout, float* inp, float* att,
                            int B, int T, int C, int NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 < T; t2++) { // ADJUSTED! this was t2 <= t (see note on function)
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += scale * local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += query_t[i] * key_t2[i]
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2];
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2];
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels
// the forward pass that is the sequence [permute, sgemm, softmax, sgemm, unpermute]

// C = 768
// NH = 12
// HS = 64

//#define C 768
//#define T 1024
//#define HS 64
//#define NH 12

// each threadblock computes a single row of q
// we need T blocks to compute a single q
// need B * T * NH to compute the whole attention

// -FLT_MAX
// fmaxf
// expf

// blockDim.x = NH
// blockDim.y = T
// blockDim.z = B
// inp is (B, T, 3, NH, HS)
// out is (B, T, NH, HS)
// l is (B, T, NH)


// input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
// preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
// that holds the pre-attention and post-attention scores (used in backward)
// output is (B, T, C)
// attention is the only layer that mixes information across time
// every other operation is applied at every (b,t) position independently
// (and of course, no layer mixes information across batch)
__global__ void flash_attention_forward_kernel0(float* out, float* inp, float* l,
                                int B, int T, int NH, int HS) {
// each threadblock computes a single row of o
// each threadblock use 1 thread, computing a single row of o
// todo: multiply all elements of preatt elementwise by scale
//offset for q for this block

    int q_offset = blockIdx.z * T * 3 * NH * HS + blockIdx.y * 3 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int k_offset = blockIdx.z * T * 3 * NH * HS +          0 * 3 * NH * HS + 1 * NH * HS + blockIdx.x * HS;
    int v_offset = blockIdx.z * T * 3 * NH * HS +          0 * 3 * NH * HS + 2 * NH * HS + blockIdx.x * HS;
    int o_offset = blockIdx.z * T * 1 * NH * HS + blockIdx.y * 1 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int l_offset = blockIdx.z * T * NH + blockIdx.y * NH + blockIdx.x;
    //printf("NH = %d, T = %d, B = %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    float* gQ = &inp[q_offset];
    float* gK = &inp[k_offset];
    float* gV = &inp[v_offset];
    float* gO = &out[o_offset];
    float* gL = &l[l_offset];
    // HS = 64
    float rQ[64] = {0.0f};
    float rO[64] = {0.0f};
    float x = 0.0f;
    float m_old = -FLT_MAX;
    float m = -FLT_MAX;
    float d_old = 0.0f;
    float d = 0.0f;

    // load q into register
    for (int i=0; i < HS; i++){
        rQ[i] = gQ[i];
//        if (blockIdx.y == 1)
//            printf("q is %f\n", gQ[i]);
    }
    //masked softmax, up to blockIdx.y-th kv
    for (int t = 0; t <= blockIdx.y; t++) {
        // compute x_i
        x = 0.0f;
        for (int i = 0; i < HS; i++) {
        // =========================
//            if (blockIdx.y == 1)
//                printf("t = %d, k is %f\n", t, gK[i]);
            x += rQ[i] * gK[i];
        }
        x *= 1.0f / sqrtf(HS);
        // compute m_i
        m = fmaxf(m_old, x);

        // compute d_i
        d = expf(m_old - m) * d_old + expf(x-m);

        // compute o_i
        for (int i = 0; i < HS; i++) {
        // =========================
//        if (blockIdx.y == 1)
//            printf("t = %d, v is %f\n", t, gV[i]);
            rO[i] = expf(m_old - m) * d_old/d * rO[i] + expf(x-m)/d * gV[i];
        }


//        if (blockIdx.y >=64 && blockIdx.y < 68 && t == 64) {
//            printf("kernel 0: t = 64, i = %d, x = %f, m = %f, m_old = %f, l = %f, l_old = %f,  p = %f\n", blockIdx.y, x, m, m_old, d, d_old, expf(x-m));
//        }

        //each block computes a single row of m
//        if (blockIdx.y >=64 && blockIdx.y < 68 && t == 63) {
//            printf("kernel 0: t = 63, i = %d, m[i] = %f, l[i] = %f\n", blockIdx.y, m, d);
//        }
////        //each block computes a single row of m
//        if (blockIdx.y >= 64 && blockIdx.y < 68 && t == blockIdx.y) {
//            printf("kernel 0: t = 127, i = %d, m[i] = %f, l[i] = %f\n", blockIdx.y, m, d);
//
//        }

        //update constants
        m_old = m;
        d_old = d;

        //update k,v offset
        gK += 3 * NH * HS;
        gV += 3 * NH * HS;
    }

    gL[0] = logf(d) + m;

    //printf("kernel 0: address = %d, i = %d, l = %f\n", l_offset, blockIdx.y, logf(d) + m);



    // write vaccum to global memory
    for (int i=0; i < HS; i++){
//        if (blockIdx.y == 1)
//            printf("o is %f\n", rO[i]);

        gO[i] = rO[i];
    }

}

#define T_r 64
#define gQ(i,j) gQ[(i) * 3 * NH * HS + (j)]
#define gK(i,j) gK[(i) * 3 * NH * HS + (j)]
#define gV(i,j) gV[(i) * 3 * NH * HS + (j)]
#define gO(i,j) gO[(i) * 1 * NH * HS + (j)]
#define gL(i) gL[(i) * NH]
#define gD(i) gD[(i) * NH]
#define sQ(i,j) sQ[(i) * HS + (j)]
#define sK(i,j) sK[(i) * HS + (j)]
#define sV(i,j) sV[(i) * HS + (j)]
#define sP(i,j) sP[(i) * HS + (j)]
#define FLOAT4(addr) reinterpret_cast<float4*>(&(addr))[0]
__global__ __launch_bounds__(256)
void flash_attention_forward_kernel1(float* out, float* inp, float* l,
                                int B, int T, int NH, int HS) {
// blockDim.x = NH
// blockDim.y = T
// blockDim.z = B
// inp is (B, T, 3, NH, HS)
// out is (B, T, NH, HS)
// l is (B, T, NH)

// we use 256 threads = 8 warps in each threadblock
// we use 64KB of shared memory for Q, K, V, P so each uses 16KB of shared memory
// 16KB of shared memory can store 16*1024/4 = 4096 floats = 64 * 64 floats
// so each threadblock computes 64 * 64 tile of O, and each warp does 8 * 64 tile of O
// following flash attention 2, we only store (m + log l) instead of (m, l) for the backward pass
// idea: if we dont use shared memory to store
    //int thread_id = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int q_global_offset = blockIdx.z * T * 3 * NH * HS + blockIdx.y * T_r * 3 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int k_global_offset = blockIdx.z * T * 3 * NH * HS +                0 * 3 * NH * HS + 1 * NH * HS + blockIdx.x * HS;
    int v_global_offset = blockIdx.z * T * 3 * NH * HS +                0 * 3 * NH * HS + 2 * NH * HS + blockIdx.x * HS;
    int o_global_offset = blockIdx.z * T * 1 * NH * HS + blockIdx.y * T_r * 1 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int l_global_offset = blockIdx.z * T * NH + blockIdx.y * T_r * NH + blockIdx.x;

    float* gQ = &inp[q_global_offset];
    float* gK = &inp[k_global_offset];
    float* gV = &inp[v_global_offset];
    float* gO = &out[o_global_offset];
    float* gL = &l[l_global_offset];

    extern __shared__ float sharedMemory[];

    float* sQ = &sharedMemory[0 * T_r * T_r];
    float* sK = &sharedMemory[1 * T_r * T_r];
    float* sV = &sharedMemory[2 * T_r * T_r];
    float* sP = &sharedMemory[3 * T_r * T_r];

    //int qkv_token_increment = 3 * NH * HS;
    int kv_tile_increment = T_r * 3 * NH * HS;
    //int o_token_increment = NH * HS;
    //int o_tile_increment = T_r * NH * HS;
    //int l_increment = NH;

    // addresses for loading data from global to shared
    // as well as for register tiling
    int warp_row = warp_id * 8;
    int thread_row = (lane_id / 16) * 4;
    int thread_col = (lane_id % 16) * 4;


    // load gQ to sQ
    for (int i = 0; i < 4; i++) {
	    FLOAT4(sQ(warp_row + thread_row + i, thread_col)) = FLOAT4(gQ(warp_row + thread_row + i, thread_col));
    }

    __syncthreads();

    // main loop
    float rQ[4] = {0.0f};
    float rK[4] = {0.0f};
    float rV[4] = {0.0f};
    float tS[4][4] = {0.0f};
    float (&tP)[4][4] = tS;
    //float tP[4][4] = {0.0f};
    float rP[4] = {0.0f};
    float rO[4][4] = {0.0f};
    float rM_old[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    float rM[4] = {0.0f};
    float rL_old[4] = {0.0f};
    float rL[4] = {0.0f};
    // this stores sum(rP) across the half-warps
    // in order to compute rL = exp(rM_old - rM) * rL_old + sum(rP)
    float rD[4] = {0.0f};

    // now we need to put the auto regressive mask, what should it be
    // only need to check when kv_tile = blockIdx.y
    for (int kv_tile = 0; kv_tile <= blockIdx.y; kv_tile++) {
//        if (threadIdx.x==0) {
//            printf("HIHIHIHELO");
//        }
        // load gK to sK, need to first transpose
        float rK_shared[4];
        for (int i = 0; i < 4; i++) {
            FLOAT4(rK_shared[0]) = FLOAT4(gK(warp_row + thread_row + i, thread_col));

            for (int j=0; j < 4; j++) {
//                if (threadIdx.x ==0 && i==0) {
//                    printf("j is %d, rK_shared is %f\n", j, rK_shared[j]);
//                }
                sK(thread_col + j, warp_row + thread_row + i) = rK_shared[j];
            }
        }

        // load gV to sV
        for (int i = 0; i < 4; i++) {
	        FLOAT4(sV(warp_row + thread_row + i, thread_col)) = FLOAT4(gV(warp_row + thread_row + i, thread_col));
        }

        __syncthreads();

//        if (threadIdx.x==0) {
//            for (int i=0;i<64;i++) {
//                for (int j=0;j<64;j++) {
//                    printf("%.2f ", sK(i,j));
//                }
//                printf("\n");
//            }
//        }

        //
        // compute rS
        //

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                tS[i][j] = 0;
            }
        }

        for (int k_fragment = 0; k_fragment < HS; k_fragment++) {
//            for (int i = 0; i < 4; i++) {
//                printf("%f ", rQ[i]);
//            }


            for (int i = 0; i < 4; i++) {
                //printf("ORIGINAL address in sQ is %d, sQ is %f, rQ is %f\n", (warp_row + thread_row + i) * 64 + k_fragment, sQ(warp_row + thread_row + i, k_fragment), rQ[i]);
                rQ[i] = sQ(warp_row + thread_row + i, k_fragment);
                //printf("UPDATE address in sQ is %d, sQ is %f, rQ is %f\n", (warp_row + thread_row + i) * 64 + k_fragment, sQ(warp_row + thread_row + i, k_fragment), rQ[i]);
                //rQ[i] = 0;
                rK[i] = sK(k_fragment, thread_col + i);

//                if (threadIdx.x ==0) {
//                        printf("k_fragment = %d, i = %d, rQ = %f, rK = %f \n", k_fragment, i, rQ[i], rK[i]);
//                }
            }


            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (blockIdx.y == kv_tile && warp_row + thread_row + i < thread_col + j) {
                            tS[i][j] = -FLT_MAX;
                        } else {
                            tS[i][j] += rQ[i] * rK[j];
                        }

//                    if (threadIdx.x == 0 and k_fragment==HS-1) {
//                        printf("i = %d, j= %d, tS[i][j] = %f \n", i, j, tS[i][j]);
//                    }
                }
            }
        }

        //rescale preatt by 1/sqrt(HS)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (tS[i][j] != -FLT_MAX) {
                    tS[i][j] *= 1.0f / sqrtf(HS);
                }
            }
        }

        //
        // compute m
        //
        unsigned mask = (lane_id < 16) ? 0xFFFF : 0xFFFF0000; // Mask for the two halves
        int thread_id_to_read_from = (lane_id < 16) ? 0 : 16; // Lane to read from
        // compute m


        // inter-thread reduction
        for (int i = 0; i < 4; i++) {
            rM[i] = rM_old[i];
            for (int j = 0; j < 4;j++) {
                rM[i] = fmaxf(rM[i], tS[i][j]);
            }
        }

        //inter-warp reduction
        for (int i=0; i < 4; i++) {
            for (int offset = 8; offset > 0; offset /= 2) {
               rM[i] = fmaxf(rM[i], __shfl_down_sync(mask, rM[i], offset));
            }
        }

        // now threads 0, 16 have the correct m[i],
        // so we broadcast m back to the other lanes in the half warp
        for (int i=0; i<4; i++) {
            rM[i] = __shfl_sync(mask, rM[i], thread_id_to_read_from);
        }

        //
        // compute P
        //
        for (int i=0;i<4;i++) {
            for (int j=0;j<4;j++){
                tP[i][j] = expf(tS[i][j] - rM[i]);
            }
        }

        //store to sP
        for (int i=0;i<4;i++) {
            for (int j=0;j<4;j++) {
                sP(warp_row + thread_row + i , thread_col + j) = tP[i][j];
            }
        }

        //
        // compute l
        //

        // rescale l and also reset rD to 0
        for (int i = 0; i < 4; i++) {
            rL[i] = expf(rM_old[i] - rM[i]) * rL_old[i];
            rD[i] = 0;
        }

        // inter-thread reduction
        for (int i = 0; i < 4; i++) {
            for (int j=0;j<4;j++){
                rD[i] += tP[i][j];
            }
        }

        //inter-warp reduction
        for (int i=0; i < 4; i++) {
            for (int offset = 8; offset > 0; offset /= 2) {
               rD[i] += __shfl_down_sync(mask, rD[i], offset);
            }
        }

        // now threads 0, 16 have the correct rD[i],
        // so we compute rL[i] and broadcast it back to the warp
        for (int i=0; i<4; i++) {
            rL[i] += rD[i];
            rL[i] = __shfl_sync(mask, rL[i], thread_id_to_read_from);
        }

        __syncthreads();

        //
        // compute O
        //

        // first rescale O by exp(m_old - m)
        for (int i=0; i<4; i++) {
            for (int j=0;j<4;j++) {
                rO[i][j] = expf(rM_old[i] - rM[i]) * rO[i][j];
            }
        }
        // add PV to rO
        for (int k_fragment = 0; k_fragment < HS; k_fragment++) {

            for (int i=0; i<4; i++) {
                rP[i] = sP(warp_row + thread_row + i, k_fragment);
                rV[i] = sV(k_fragment, thread_col + i);
            }

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    rO[i][j] += rP[i] * rV[j];
                }
            }
        }

        // update m and l
        for (int i = 0; i < 4; i++) {
            rM_old[i] = rM[i];
            rL_old[i] = rL[i];
        }

        gK += kv_tile_increment;
        gV += kv_tile_increment;
        __syncthreads();
    }


    //rescale rO
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            rO[i][j] /= rL[i];
        }
    }

//    if (lane_id == 0 || lane_id == 16) {
//        for (int i=0;i<4;i++) {
//            printf("kernel 1: i = %d, l = %f\n", blockIdx.y * 64 + warp_row + thread_row + i, rM[i] + logf(rL[i]));
//        }
//    }
    // store l back to gL
    if (lane_id == 0 || lane_id == 16) {
        for (int i = 0; i < 4; i++) {
            //printf("kernel 1: address = %d, i = %d, l = %f\n", l_global_offset + (warp_row + thread_row + i) * NH, blockIdx.y * 64 + warp_row + thread_row + i, rM[i] + logf(rL[i]));
            gL(warp_row + thread_row + i) = rM[i] + logf(rL[i]);
        }
    }

    // store rO to gO
    for (int i=0; i < 4; i++) {
        FLOAT4(gO(warp_row + thread_row + i, thread_col)) = FLOAT4(rO[i][0]);
//        for (int j=0;j<4;j++) {
//            gO(warp_row + thread_row + i, thread_col + j) = rO[i][j];
//        }
    }

}

#undef sP


__global__ __launch_bounds__(256)
void flash_attention_forward_kernel2(float* out, float* inp, float* l,
                                int B, int T, int NH, int HS) {
// blockDim.x = NH
// blockDim.y = T
// blockDim.z = B
// inp is (B, T, 3, NH, HS)
// out is (B, T, NH, HS)
// l is (B, T, NH)

// we use 256 threads = 8 warps in each threadblock
// we use 48KB of shared memory for Q, K, V so each uses 16KB of shared memory
// 16KB of shared memory can store 16*1024/4 = 4096 floats = 64 * 64 floats
// so each threadblock computes 64 * 64 tile of O, and each warp does 8 * 64 tile of O
// following flash attention 2, we only store (m + log l) instead of (m, l) for the backward pass
// idea: if we dont use shared memory to store
    //int thread_id = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int q_global_offset = blockIdx.z * T * 3 * NH * HS + blockIdx.y * T_r * 3 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int k_global_offset = blockIdx.z * T * 3 * NH * HS +                0 * 3 * NH * HS + 1 * NH * HS + blockIdx.x * HS;
    int v_global_offset = blockIdx.z * T * 3 * NH * HS +                0 * 3 * NH * HS + 2 * NH * HS + blockIdx.x * HS;
    int o_global_offset = blockIdx.z * T * 1 * NH * HS + blockIdx.y * T_r * 1 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int l_global_offset = blockIdx.z * T * NH + blockIdx.y * T_r * NH + blockIdx.x;

    float* gQ = &inp[q_global_offset];
    float* gK = &inp[k_global_offset];
    float* gV = &inp[v_global_offset];
    float* gO = &out[o_global_offset];
    float* gL = &l[l_global_offset];

    extern __shared__ float sharedMemory[];

    float* sQ = &sharedMemory[0 * T_r * T_r];
    float* sK = &sharedMemory[1 * T_r * T_r];
    float* sV = &sharedMemory[2 * T_r * T_r];

    //int qkv_token_increment = 3 * NH * HS;
    int kv_tile_increment = T_r * 3 * NH * HS;
    //int o_token_increment = NH * HS;
    //int o_tile_increment = T_r * NH * HS;
    //int l_increment = NH;

    // addresses for loading data from global to shared
    // as well as for register tiling
    int warp_row = warp_id * 8;
    int thread_row = (lane_id / 16) * 4;
    int thread_col = (lane_id % 16) * 4;


    // load gQ to sQ
    for (int i = 0; i < 4; i++) {
	    FLOAT4(sQ(warp_row + thread_row + i, thread_col)) = FLOAT4(gQ(warp_row + thread_row + i, thread_col));
    }

    __syncthreads();

    // main loop
    float rQ[4] = {0.0f};
    float rK[4] = {0.0f};
    float rV[4] = {0.0f};
    float tS[4][4] = {0.0f};
    float (&tP)[4][4] = tS;
    //float tP[4][4] = {0.0f};
    float rP[4] = {0.0f};
    float rO[4][4] = {0.0f};
    float rM_old[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    float rM[4] = {0.0f};
    float rL_old[4] = {0.0f};
    float rL[4] = {0.0f};
    // this stores sum(rP) across the half-warps
    // in order to compute rL = exp(rM_old - rM) * rL_old + sum(rP)
    float rD[4] = {0.0f};

    // now we need to put the auto regressive mask, what should it be
    // only need to check when kv_tile = blockIdx.y
    for (int kv_tile = 0; kv_tile <= blockIdx.y; kv_tile++) {
//        if (threadIdx.x==0) {
//            printf("HIHIHIHELO");
//        }
        // load gK to sK, need to first transpose
        float rK_shared[4];
        for (int i = 0; i < 4; i++) {
            FLOAT4(rK_shared[0]) = FLOAT4(gK(warp_row + thread_row + i, thread_col));

            for (int j=0; j < 4; j++) {
//                if (threadIdx.x ==0 && i==0) {
//                    printf("j is %d, rK_shared is %f\n", j, rK_shared[j]);
//                }
                sK(thread_col + j, warp_row + thread_row + i) = rK_shared[j];
            }
        }

        __syncthreads();

        // load gV to sV
        for (int i = 0; i < 4; i++) {
	        FLOAT4(sV(warp_row + thread_row + i, thread_col)) = FLOAT4(gV(warp_row + thread_row + i, thread_col));
        }



//        if (threadIdx.x==0) {
//            for (int i=0;i<64;i++) {
//                for (int j=0;j<64;j++) {
//                    printf("%.2f ", sK(i,j));
//                }
//                printf("\n");
//            }
//        }

        //
        // compute rS
        //

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                tS[i][j] = 0;
            }
        }

        for (int k_fragment = 0; k_fragment < HS; k_fragment++) {
//            for (int i = 0; i < 4; i++) {
//                printf("%f ", rQ[i]);
//            }


            for (int i = 0; i < 4; i++) {
                //printf("ORIGINAL address in sQ is %d, sQ is %f, rQ is %f\n", (warp_row + thread_row + i) * 64 + k_fragment, sQ(warp_row + thread_row + i, k_fragment), rQ[i]);
                rQ[i] = sQ(warp_row + thread_row + i, k_fragment);
                //printf("UPDATE address in sQ is %d, sQ is %f, rQ is %f\n", (warp_row + thread_row + i) * 64 + k_fragment, sQ(warp_row + thread_row + i, k_fragment), rQ[i]);
                //rQ[i] = 0;
                rK[i] = sK(k_fragment, thread_col + i);

//                if (threadIdx.x ==0) {
//                        printf("k_fragment = %d, i = %d, rQ = %f, rK = %f \n", k_fragment, i, rQ[i], rK[i]);
//                }
            }


            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (blockIdx.y == kv_tile && warp_row + thread_row + i < thread_col + j) {
                            tS[i][j] = -FLT_MAX;
                        } else {
                            tS[i][j] += rQ[i] * rK[j];
                        }

//                    if (threadIdx.x == 0 and k_fragment==HS-1) {
//                        printf("i = %d, j= %d, tS[i][j] = %f \n", i, j, tS[i][j]);
//                    }
                }
            }
        }

        //rescale preatt by 1/sqrt(HS)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (tS[i][j] != -FLT_MAX) {
                    tS[i][j] *= 1.0f / sqrtf(HS);
                }
            }
        }

        //
        // compute m
        //
        unsigned mask = (lane_id < 16) ? 0xFFFF : 0xFFFF0000; // Mask for the two halves
        int lane_id_to_read_from = (lane_id < 16) ? 0 : 16; // Lane to read from
        // compute m


        // inter-thread reduction
        for (int i = 0; i < 4; i++) {
            rM[i] = rM_old[i];
            for (int j = 0; j < 4;j++) {
                rM[i] = fmaxf(rM[i], tS[i][j]);
            }
        }

        //inter-warp reduction
        for (int i=0; i < 4; i++) {
            for (int offset = 8; offset > 0; offset /= 2) {
               rM[i] = fmaxf(rM[i], __shfl_down_sync(mask, rM[i], offset));
            }
        }

        // now threads 0, 16 have the correct m[i],
        // so we broadcast m back to the other lanes in the half warp
        for (int i=0; i<4; i++) {
            rM[i] = __shfl_sync(mask, rM[i], lane_id_to_read_from);
        }

        //
        // compute P
        //
        for (int i=0;i<4;i++) {
            for (int j=0;j<4;j++){
                tP[i][j] = expf(tS[i][j] - rM[i]);
            }
        }

        //store to sP
//        for (int i=0;i<4;i++) {
//            for (int j=0;j<4;j++) {
//                sP(warp_row + thread_row + i , thread_col + j) = tP[i][j];
//            }
//        }

        //
        // compute l
        //

        // rescale l and also reset rD to 0
        for (int i = 0; i < 4; i++) {
            rL[i] = expf(rM_old[i] - rM[i]) * rL_old[i];
            rD[i] = 0;
        }

        // inter-thread reduction
        for (int i = 0; i < 4; i++) {
            for (int j=0;j<4;j++){
                rD[i] += tP[i][j];
            }
        }

        //inter-warp reduction
        for (int i=0; i < 4; i++) {
            for (int offset = 8; offset > 0; offset /= 2) {
               rD[i] += __shfl_down_sync(mask, rD[i], offset);
            }
        }

        // now threads 0, 16 have the correct rD[i],
        // so we compute rL[i] and broadcast it back to the warp
        for (int i=0; i<4; i++) {
            rL[i] += rD[i];
            rL[i] = __shfl_sync(mask, rL[i], lane_id_to_read_from);
        }

        __syncthreads();

        //
        // compute O
        //

        // first rescale O by exp(m_old - m)
        for (int i=0; i<4; i++) {
            for (int j=0;j<4;j++) {
                rO[i][j] = expf(rM_old[i] - rM[i]) * rO[i][j];
            }
        }
        // add PV to rO
//        for (int k_fragment = 0; k_fragment < 64; k_fragment++) {
//
//            for (int i=0; i<4; i++) {
//                // which thread
//                // (lane_id / 16) * 16  + k_fragment / 4
//                // which i in tP
//                // i
//                // which j in tP
//                // k_fragment % 4
//                rP[i] = __shfl_sync(mask, tP[i][k_fragment % 4], (lane_id >> 4) * 16  + (k_fragment >> 2));
//                rV[i] = sV(k_fragment, thread_col + i);
//            }
//
//            for (int i = 0; i < 4; i++) {
//                for (int j = 0; j < 4; j++) {
//                    rO[i][j] += rP[i] * rV[j];
//                }
//            }
//        }
        for (int l = 0; l < 16; l++) {

            for (int k=0; k < 4; k++) {
                for (int i=0; i<4; i++) {
                    // which thread
                    // (lane_id / 16) * 16  + k_fragment / 4
                    // which i in tP
                    // i
                    // which j in tP
                    // k_fragment % 4
                    rP[i] = __shfl_sync(mask, tP[i][k], (lane_id >> 4) * 16  + ((l*4 + k) >> 2));
                    rV[i] = sV(l * 4 + k, thread_col + i);
                }

                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        rO[i][j] += rP[i] * rV[j];
                    }
                }

            }

        }


        // update m and l
        for (int i = 0; i < 4; i++) {
            rM_old[i] = rM[i];
            rL_old[i] = rL[i];
        }

        gK += kv_tile_increment;
        gV += kv_tile_increment;
        __syncthreads();
    }


    //rescale rO
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            rO[i][j] /= rL[i];
        }
    }

//    if (lane_id == 0 || lane_id == 16) {
//        for (int i=0;i<4;i++) {
//            printf("kernel 1: i = %d, l = %f\n", blockIdx.y * 64 + warp_row + thread_row + i, rM[i] + logf(rL[i]));
//        }
//    }
    // store l back to gL
    if (lane_id == 0 || lane_id == 16) {
        for (int i = 0; i < 4; i++) {
            //printf("kernel 1: address = %d, i = %d, l = %f\n", l_global_offset + (warp_row + thread_row + i) * NH, blockIdx.y * 64 + warp_row + thread_row + i, rM[i] + logf(rL[i]));
            gL(warp_row + thread_row + i) = rM[i] + logf(rL[i]);
        }
    }

    // store rO to gO
    for (int i=0; i < 4; i++) {
        FLOAT4(gO(warp_row + thread_row + i, thread_col)) = FLOAT4(rO[i][0]);
//        for (int j=0;j<4;j++) {
//            gO(warp_row + thread_row + i, thread_col + j) = rO[i][j];
//        }
    }

}

#undef T_r
#undef sQ
#undef sK
#undef sV

#define TILE_SIZE 128
#define HEAD_SIZE 64
#define sQ(i,j) sQ[(i) + (j) * TILE_SIZE]
#define sK(i,j) sK[(i) * TILE_SIZE + (j)]
#define sV(i,j) sV[(i) * HEAD_SIZE + (j)]


__global__ __launch_bounds__(256)
void flash_attention_forward_kernel3(float* out, float* inp, float* l,
                                int B, int T, int NH, int HS) {
// blockDim.x = NH
// blockDim.y = T
// blockDim.z = B
// inp is (B, T, 3, NH, HS)
// out is (B, T, NH, HS)
// l is (B, T, NH)

// we use 256 threads = 8 warps in each threadblock
// we use 96KB of shared memory for Q, K, V so each uses 32KB of shared memory
// 32KB of shared memory can store 32*1024/4 = 8192 floats = 128 * 64 floats
// so each threadblock computes 128 * 64 tile of O, and each warp does 16 * 64 tile of O
// following flash attention 2, we only store (m + log l) instead of (m, l) for the backward pass
// idea: if we dont use shared memory to store
    //int thread_id = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int q_global_offset = blockIdx.z * T * 3 * NH * HS + blockIdx.y * TILE_SIZE * 3 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int k_global_offset = blockIdx.z * T * 3 * NH * HS +                      0 * 3 * NH * HS + 1 * NH * HS + blockIdx.x * HS;
    int v_global_offset = blockIdx.z * T * 3 * NH * HS +                      0 * 3 * NH * HS + 2 * NH * HS + blockIdx.x * HS;
    int o_global_offset = blockIdx.z * T * 1 * NH * HS + blockIdx.y * TILE_SIZE * 1 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int l_global_offset = blockIdx.z * T * NH + blockIdx.y * TILE_SIZE * NH + blockIdx.x;

    float* gQ = &inp[q_global_offset];
    float* gK = &inp[k_global_offset];
    float* gV = &inp[v_global_offset];
    float* gO = &out[o_global_offset];
    float* gL = &l[l_global_offset];

    extern __shared__ float sharedMemory[];

    float* sQ = &sharedMemory[0 * TILE_SIZE * 64];
    float* sK = &sharedMemory[1 * TILE_SIZE * 64];
    float* sV = &sharedMemory[2 * TILE_SIZE * 64];

    int tile_increment = TILE_SIZE * 3 * NH * HS;

    // addresses for loading data from global to shared
    // as well as for register tiling
    int warp_row = warp_id * 16;
    int thread_row = (lane_id / 16) * 4;
    int thread_col = (lane_id % 16) * 4;


    // load gQ to sQ

    float rQ_shared[2][4];

    for (int i = 0; i < 4; i++) {
        FLOAT4(rQ_shared[0][0]) = FLOAT4(gQ(warp_row + thread_row + i, thread_col));
        FLOAT4(rQ_shared[1][0]) = FLOAT4(gQ(warp_row + thread_row + i + 8, thread_col));

        for (int j=0; j < 4; j++) {
            sQ(warp_row + thread_row + i, thread_col + j) = rQ_shared[0][j];
            sQ(warp_row + thread_row + i + 8, thread_col + j) = rQ_shared[1][j];
        }

    }

    // main loop
    float rQ[8] = {0.0f};
    float rK[8] = {0.0f};
    float rV[4] = {0.0f};
    float tS[8][8] = {0.0f};
    float (&tP)[8][8] = tS;
    //float tP[4][4] = {0.0f};
    float rP[8] = {0.0f};
    float rO[8][4] = {0.0f};
    float rM_old[8] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    float rM[8] = {0.0f};
    float rL_old[8] = {0.0f};
    float rL[8] = {0.0f};
    // this stores sum(rP) across the half-warps
    // in order to compute rL = exp(rM_old - rM) * rL_old + sum(rP)
    float rD[8] = {0.0f};
    // For auto regressive mask, need to check when kv_tile = blockIdx.y
    for (int tile = 0; tile <= blockIdx.y; tile++) {

        // load gK to sK, need to first transpose
        float rK_shared[2][4];

        for (int i = 0; i < 4; i++) {
            FLOAT4(rK_shared[0][0]) = FLOAT4(gK(warp_row + thread_row + i, thread_col));
            FLOAT4(rK_shared[1][0]) = FLOAT4(gK(warp_row + thread_row + i + 8, thread_col));

            for (int j=0; j < 4; j++) {
                sK(thread_col + j, warp_row + thread_row + i) = rK_shared[0][j];
                sK(thread_col + j, warp_row + thread_row + i + 8) = rK_shared[1][j];
            }

        }

        // load gV to sV
        for (int i = 0; i < 4; i++) {
            FLOAT4(sV(warp_row + thread_row + i, thread_col)) = FLOAT4(gV(warp_row + thread_row + i, thread_col));
            FLOAT4(sV(warp_row + thread_row + i + 8, thread_col)) = FLOAT4(gV(warp_row + thread_row + i + 8, thread_col));
        }

        __syncthreads();



        //
        // compute rS
        //

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                tS[i][j] = 0;
            }
        }

        for (int k_fragment = 0; k_fragment < HEAD_SIZE; k_fragment++) {

            FLOAT4(rQ[0]) = FLOAT4(sQ(warp_row + thread_row, k_fragment));
            FLOAT4(rQ[4]) = FLOAT4(sQ(warp_row + thread_row + 8, k_fragment));
            FLOAT4(rK[0]) = FLOAT4(sK(k_fragment, thread_col));
            FLOAT4(rK[4]) = FLOAT4(sK(k_fragment, thread_col + 64));


            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (tile == blockIdx.y  && warp_row + thread_row + i < thread_col + j) {
                        tS[i][j] = -FLT_MAX;
                    } else {
                        tS[i][j] += rQ[i] * rK[j];
                    }

                    if (tile == blockIdx.y  && warp_row + thread_row + i + 8 < thread_col + j) {
                        tS[i + 4][j] = -FLT_MAX;
                    } else {
                        tS[i + 4][j] += rQ[i + 4] * rK[j];
                    }

                    if (tile == blockIdx.y  && warp_row + thread_row + i < thread_col + j + 64) {
                        tS[i][j+4] = -FLT_MAX;
                    } else {
                        tS[i][j+4] += rQ[i] * rK[j+4];
                    }

                    if (tile == blockIdx.y  && warp_row + thread_row + i + 8 < thread_col + j + 64) {
                        tS[i+4][j+4] = -FLT_MAX;
                    } else {
                        tS[i+4][j+4] += rQ[i+4] * rK[j+4];
                    }
                }
            }
        }

        //rescale preatt by 1/sqrt(HS)
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (tS[i][j] != -FLT_MAX) {
                    tS[i][j] *= 1.0f / sqrtf(HS);
                }
            }
        }

        //
        // compute m
        //
        unsigned mask = (lane_id < 16) ? 0xFFFF : 0xFFFF0000; // Mask for the two halves
        int lane_id_to_read_from = (lane_id < 16) ? 0 : 16; // Lane to read from
        // compute m


        // inter-thread reduction
        for (int i = 0; i < 8; i++) {
            rM[i] = rM_old[i];
            for (int j = 0; j < 8;j++) {
                rM[i] = fmaxf(rM[i], tS[i][j]);
            }
        }

        //inter-warp reduction
        for (int i=0; i < 8; i++) {
            for (int offset = 8; offset > 0; offset /= 2) {
               rM[i] = fmaxf(rM[i], __shfl_down_sync(mask, rM[i], offset));
            }
        }

        // now threads 0, 16 have the correct m[i],
        // so we broadcast m back to the other lanes in the half warp
        for (int i=0; i<8; i++) {
            rM[i] = __shfl_sync(mask, rM[i], lane_id_to_read_from);
        }

        //
        // compute P
        //
        for (int i=0;i<8;i++) {
            for (int j=0;j<8;j++){
                tP[i][j] = expf(tS[i][j] - rM[i]);
            }
        }

        //store to sP


        //
        // compute l
        //

        // rescale l and also reset rD to 0
        for (int i = 0; i < 8; i++) {
            rL[i] = expf(rM_old[i] - rM[i]) * rL_old[i];
            rD[i] = 0;
        }

        // inter-thread reduction
        for (int i = 0; i < 8; i++) {
            for (int j=0;j<8;j++){
                rD[i] += tP[i][j];
            }
        }

        //inter-warp reduction
        for (int i=0; i < 8; i++) {
            for (int offset = 8; offset > 0; offset /= 2) {
               rD[i] += __shfl_down_sync(mask, rD[i], offset);
            }
        }

        // now threads 0, 16 have the correct rD[i],
        // so we compute rL[i] and broadcast it back to the warp
        for (int i=0; i<8; i++) {
            rL[i] += rD[i];
            rL[i] = __shfl_sync(mask, rL[i], lane_id_to_read_from);
        }



        //
        // compute O
        //

        // first rescale O by exp(m_old - m)
        for (int i=0; i<8; i++) {
            for (int j=0;j<4;j++) {
                rO[i][j] = expf(rM_old[i] - rM[i]) * rO[i][j];
            }
        }

        // add PV to rO

        // We use warp shuffling to directly load data to each fragment from tP to compute the outer product tO.
        // To do this, there is some array indexing involving the modulo operator for tP to compute the lane id that we want to load data from.
        // For some reason, in this case the compiler will put tP into local memory, causing register spillage,
        // even though the array indexing can be computed at compile time.
        // To resolve this, we use nested for loops to remove the use of modulo operator.
        for (int h = 0; h < 2; h++) {
            for (int l = 0; l < 16; l++) {
                for (int k = 0; k < 4; k++) {
                    // position is h * 64 + l * 4 + k
                    for (int i=0;i<4;i++) {
                        rP[i] = __shfl_sync(mask, tP[i][k + h * 4], (lane_id /16) * 16  + ((l * 4 + k) / 4));
                        rP[i + 4] = __shfl_sync(mask, tP[i + 4][k + h * 4], (lane_id /16) * 16  + ((l * 4 + k) / 4));
                        rV[i] = sV(h * 64 + l * 4 + k, thread_col + i);
                    }

                    for (int i = 0; i < 8; i++) {
                        for (int j = 0; j < 4; j++) {
                            rO[i][j] += rP[i] * rV[j];
                        }
                    }
                }
            }
        }


        // update m and l
        for (int i = 0; i < 8; i++) {
            rM_old[i] = rM[i];
            rL_old[i] = rL[i];
        }

        gK += tile_increment;
        gV += tile_increment;
        __syncthreads();
    }


    //rescale rO
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            rO[i][j] /= rL[i];
        }
    }


    // store l back to gL
    if (lane_id == 0 || lane_id == 16) {
        for (int i = 0; i < 4; i++) {
            //printf("kernel 1: address = %d, i = %d, l = %f\n", l_global_offset + (warp_row + thread_row + i) * NH, blockIdx.y * 64 + warp_row + thread_row + i, rM[i] + logf(rL[i]));
            gL(warp_row + thread_row + i) = rM[i] + logf(rL[i]);
            gL(warp_row + thread_row + 8 + i) = rM[i + 4] + logf(rL[i + 4]);
        }
    }

    // store rO to gO
    for (int i=0; i < 4; i++) {
        FLOAT4(gO(warp_row + thread_row + i, thread_col)) = FLOAT4(rO[i][0]);
        FLOAT4(gO(warp_row + thread_row + 8 + i, thread_col)) = FLOAT4(rO[i+4][0]);
    }

}

#undef TILE_SIZE
#undef HEAD_SIZE
#undef sQ
#undef sK
#undef sV

__global__ void flash_attention_backward_kernel0(float* dinp, float* inp, float* dout, float* out, float* l,
                                int B, int T, int NH, int HS) {
// inp/dinp are (B, T, 3C), (B, T, 3, NH, HS) Q,K,V
//
// att/datt/dpreatt are (B, NH, T, T)
// dout is (B, T, C)
// l is (B, T, NH)
// d is (B, T, NH)
// blockDim.x = NH
// blockDim.y = T
// blockDim.z = B

// each threadblock use 1 thread, computing a single row of dq, dk and dv
    // offset for the dq,dk,dv rows that this
    int q_offset_start = blockIdx.z * T * 3 * NH * HS + 0 * 3 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int k_offset_start = blockIdx.z * T * 3 * NH * HS + 0 * 3 * NH * HS + 1 * NH * HS + blockIdx.x * HS;
    int v_offset_start = blockIdx.z * T * 3 * NH * HS + 0 * 3 * NH * HS + 2 * NH * HS + blockIdx.x * HS;
    int o_offset_start = blockIdx.z * T * 1 * NH * HS + 0 * 1 * NH * HS + 0 * NH * HS + blockIdx.x * HS;

    // offset for the q,k,v of the corresponding head
    int q_offset_current = blockIdx.z * T * 3 * NH * HS + blockIdx.y * 3 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int k_offset_current = blockIdx.z * T * 3 * NH * HS + blockIdx.y * 3 * NH * HS + 1 * NH * HS + blockIdx.x * HS;
    int v_offset_current = blockIdx.z * T * 3 * NH * HS + blockIdx.y * 3 * NH * HS + 2 * NH * HS + blockIdx.x * HS;
    int o_offset_current = blockIdx.z * T * 1 * NH * HS + blockIdx.y * 1 * NH * HS + 0 * NH * HS + blockIdx.x * HS;

    // following flashattention 2, we only store (log (L) + m) instead of L, m
    int l_offset_start = blockIdx.z * T * NH + 0 * NH + blockIdx.x;
    int l_offset_current = blockIdx.z * T * NH + blockIdx.y * NH + blockIdx.x;

    int qkv_T_increment = 3 * NH * HS;
    int o_T_increment = NH * HS;
    int l_T_increment = NH;

    float* gQ = &inp[0];
    float* gK = &inp[0];
    float* gV = &inp[0];
    float* gO = &out[0];
    float* gdO = &dout[0];

    float* gdQ = &dinp[0];
    float* gdK = &dinp[0];
    float* gdV = &dinp[0];



    // HS = 64
    float rdQ[64] = {0.0f};
    float rdK[64] = {0.0f};
    float rdV[64] = {0.0f};



    // logf
    // dv
    // loop over T
    for (int j=0; j < T; j++) {

        if (j >= blockIdx.y) {
            // compute the exponential term
            float qk = 0;
            float ll = l[l_offset_start + j * l_T_increment];
            // inner product for QK^T
            for (int k=0; k < HS; k++){
                qk += gQ[q_offset_start + j * qkv_T_increment + k] * gK[k_offset_current + k];
            }
            float e = expf(qk / sqrtf(HS) - ll);

            for (int i=0; i < HS;i++) {
                rdV[i] += e * gdO[o_offset_start + j * o_T_increment + i];
            }
        }
    }

    for (int i=0;i<HS;i++){
        //printf("checking inside kernel %d %f\n", i, rdV[i]);
        gdV[v_offset_current + i] = rdV[i];
    }


    // dk
    for (int j=0; j < T; j++) {
        if (j >= blockIdx.y) {
            // exp(qk^T-m)/L
            float qk = 0;
            float ll = l[l_offset_start + j * l_T_increment];
            for (int k=0;k<HS;k++) {
                qk += gQ[q_offset_start + j * qkv_T_increment + k] * gK[k_offset_current + k];
            }
            float e = expf(qk / sqrtf(HS) - ll);

            // dOV^T/
            float dov = 0;
            for (int k=0; k<HS; k++) {
                dov += gdO[o_offset_start + j * o_T_increment + k] * gV[v_offset_current + k];
            }
            // dO O^T
            float doo = 0;
            for (int k=0; k < HS; k++) {
                doo += gdO[o_offset_start + j * o_T_increment + k] * gO[o_offset_start + j * o_T_increment + k];
            }
            //printf("INSIDE KERNEL, doo = %f, dov = %f\n", doo, dov);
            for (int i=0;i< HS; i++){
                //printf("INSIDE KERNEL, i = %d, o = %f, v = %f\n", i, gV[v_offset_current + i], gO[o_offset_start + j * o_T_increment + i]);
                rdK[i] += e * (dov - doo) * gQ[q_offset_start + j * qkv_T_increment + i ];
            }
        }
    }

//    for (int i=0;i<HS;i++) {
//        printf("INSIDE KERNEL, i = %d, o = %f, v = %f\n", i, inp[2 * HS + i], out[i]);
//    }

//    for (int i=0;i<HS;i++) {
//        printf("INSIDE KERNEL, i = %d, o = %f, v = %f\n", i, inp[2 * HS + i], inp[i]);
//    }

    for (int i=0;i<HS;i++) {
        gdK[k_offset_current + i] = rdK[i] / sqrtf(HS);
    }

    // dq
    for (int j=0; j < T; j++) {
        if (j <= blockIdx.y) {
            float qk = 0;
            float ll = l[l_offset_current];
            for (int k=0;k<HS;k++) {
                qk += gQ[q_offset_current + k] * gK[k_offset_start + j * qkv_T_increment + k];
            }
            float e = expf(qk / sqrtf(HS) - ll);

            // dOV^T/
            float dov = 0;
            for (int k=0; k < HS; k++) {
                dov += gdO[o_offset_current + k] * gV[v_offset_start + j * qkv_T_increment + k];
            }
            // dO O^T
            float doo = 0;
            for (int k=0; k < HS; k++) {
                doo += gdO[o_offset_current + k] * gO[o_offset_current + k];
            }

            for (int i=0;i< HS; i++){
                rdQ[i] += e * (dov - doo) * gK[k_offset_start + j * qkv_T_increment + i ];
            }
        }

    }

    for (int i=0;i<HS;i++) {
        gdQ[q_offset_current + i] = rdQ[i] / sqrtf(HS);
    }
}



// preprocessing D = rowsum(dO * O)
__global__ void flash_attention_backward_preprocessing_kernel1(float* d, float* dout, float* out,
                                int B, int T, int NH, int HS) {
    // both dO and O are (B, T, NH, HS)
    // d is (B, T, NH)
    // each thread compute a single token for a single head
    // so we have B * T * NH
    // blockDim.x = NH
    // blockDim.y = T
    // blockDim.z = B
    int offset_o = blockIdx.z * T * NH * HS + blockIdx.y * NH * HS + blockIdx.x * HS;
    int offset_d = blockIdx.z * T * NH + blockIdx.y * NH + blockIdx.x;

    float reg_d = 0;
    for (int i =0; i < HS; i++) {
        reg_d += dout[offset_o + i] * out[offset_o + i];
    }

    d[offset_d] = reg_d;
}



#define TILE_SIZE 32
#define HEAD_SIZE 64

#define gQ(i,j) gQ[(i) * 3 * NH * HS + (j)]
#define gK(i,j) gK[(i) * 3 * NH * HS + (j)]
#define gV(i,j) gV[(i) * 3 * NH * HS + (j)]

#define gdQ(i,j) gdQ[(i) * 3 * NH * HS + (j)]
#define gdK(i,j) gdK[(i) * 3 * NH * HS + (j)]
#define gdV(i,j) gdV[(i) * 3 * NH * HS + (j)]

#define gdO(i,j) gdO[(i) * 1 * NH * HS + (j)]
#define gL(i) gL[(i) * NH]
#define gD(i) gD[(i) * NH]

#define sQ(i,j) sQ[(i) * HEAD_SIZE + (j)]
#define sK(i,j) sK[(i) * HEAD_SIZE + (j)]
#define sK_T(i,j) sK[(i) + (j) * HEAD_SIZE]
#define sV(i,j) sV[(i) * HEAD_SIZE + (j)]
#define sV_T(i,j) sV[(i) + (j) * HEAD_SIZE]

#define sdO(i,j) sdO[(i) * HEAD_SIZE + (j)]
#define sdQ(i,j) sdQ[(i) * HEAD_SIZE + (j)]
#define sP(i,j) sP[(i) * TILE_SIZE + (j)]
#define sP_T(i,j) sP[(i) + (j) * TILE_SIZE]
#define sdS(i,j) sdS[(i) * TILE_SIZE + (j)]
#define sdS_T(i,j) sdS[(i) + (j) * TILE_SIZE]
//#define FLOAT4(value) *reinterpret_cast<float4*>(&(value))[0]


__global__ void flash_attention_backward_kernel1(float* dinp, float* inp, float* dout, float* out, float* l, float* d,
                                int B, int T, int NH, int HS) {
    // inp  (B, T, 3, NH, HS)
    // out  (B, T, NH, HS)
    // dout (B, T, NH, HS)
    // l    (B, T, NH)
    // d    (B, T, NH)

    // dinp (B, T, 3, NH, HS)

    // blockDim.x = NH
    // blockDim.y = T
    // blockDim.z = B

    // load Q, K, V, dO, dQ
    // each is a 32 * 64 logical matrix, so each takes 32 * 64 * 4 / 1024 = 8KB
    // total smem size is 8*5 = 40KB
    // all matrices in shared memory is a 32 x 64 row major matrix


    // Each threadblock does a single 32 * 64
    // So we launch B * T / TILE_SIZE * NH blocks
    // we launch 128 thread per block

    int thread_id = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;


    int q_global_offset_start = blockIdx.z * T * 3 * NH * HS + 0 * 3 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int k_global_offset_start = blockIdx.z * T * 3 * NH * HS + 0 * 3 * NH * HS + 1 * NH * HS + blockIdx.x * HS;
    int v_global_offset_start = blockIdx.z * T * 3 * NH * HS + 0 * 3 * NH * HS + 2 * NH * HS + blockIdx.x * HS;
    int o_global_offset_start = blockIdx.z * T * 1 * NH * HS + 0 * 1 * NH * HS + 0 * NH * HS + blockIdx.x * HS;

    // offset for the q,k,v of the corresponding head
    int q_global_offset_current = blockIdx.z * T * 3 * NH * HS + blockIdx.y * TILE_SIZE * 3 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int k_global_offset_current = blockIdx.z * T * 3 * NH * HS + blockIdx.y * TILE_SIZE * 3 * NH * HS + 1 * NH * HS + blockIdx.x * HS;
    int v_global_offset_current = blockIdx.z * T * 3 * NH * HS + blockIdx.y * TILE_SIZE * 3 * NH * HS + 2 * NH * HS + blockIdx.x * HS;
    int o_global_offset_current = blockIdx.z * T * 1 * NH * HS + blockIdx.y * TILE_SIZE * 1 * NH * HS + 0 * NH * HS + blockIdx.x * HS;

    // following flashattention 2, we only store (log (L) + m) instead of L, m
    int ld_global_offset_start = blockIdx.z * T * NH + 0 * NH + blockIdx.x;
    int ld_global_offset_current = blockIdx.z * T * NH + blockIdx.y * TILE_SIZE * NH + blockIdx.x;

    int qkv_increment = TILE_SIZE * 3 * NH * HS;
    int o_increment = TILE_SIZE * NH * HS;
    int ld_increment = TILE_SIZE * NH;

    float* gQ = &inp[q_global_offset_current];
    float* gK = &inp[k_global_offset_current];
    float* gV = &inp[v_global_offset_current];
    float* gdO = &dout[o_global_offset_current];
    float* gL = &l[ld_global_offset_current];
    float* gD = &d[ld_global_offset_current];


    // output
    float* gdQ = &dinp[q_global_offset_current];
    float* gdK = &dinp[k_global_offset_current];
    float* gdV = &dinp[v_global_offset_current];

    extern __shared__ float sharedMemory[];

    float* sQ  = &sharedMemory[0 * TILE_SIZE * HEAD_SIZE];
    float* sK  = &sharedMemory[1 * TILE_SIZE * HEAD_SIZE];
    float* sV  = &sharedMemory[2 * TILE_SIZE * HEAD_SIZE];
    float* sdQ = &sharedMemory[3 * TILE_SIZE * HEAD_SIZE];
    float* sdO = &sharedMemory[4 * TILE_SIZE * HEAD_SIZE];
    float* sP = &sharedMemory[5 * TILE_SIZE * HEAD_SIZE];
    float* sdS = &sharedMemory[5 * TILE_SIZE * HEAD_SIZE + HEAD_SIZE * HEAD_SIZE];



    // offset for loading K, V from global to shared
    int thread_row_copy = warp_id * 8 + (lane_id / 16) * 4;
    int thread_col_copy = (lane_id % 16) * 4;

    // offset when computing 32 x 32 miccro kernel
    int thread_row_32_x_32 = warp_id * 8 + (lane_id / 16) * 4;
    int thread_col_32_x_32 = (lane_id % 16) * 2;

    // offset when computing 32 x 64 miccro kernel
    int thread_row_32_x_64 = warp_id * 8 + (lane_id / 16) * 4;
    int thread_col_32_x_64 = (lane_id % 16) * 4;

    // offset for loading back to global memory
    int thread_row_atomic_add = warp_id * 8;
    int thread_col_atomic_add = lane_id;

    float rL[4];
    float rD[4];

    float rQ[4];
    float rK[4];
    float rV[4];
    float rdO[4];
    float rP[4];
    float rdS[4];

    float tdQ[4][4] = {0.0f};
    float tdK[4][4] = {0.0f};
    float tdV[4][4] = {0.0f};

    float tS[4][2] = {0.0f};
    float (&tP)[4][2] = tS;
    float tdP[4][2] = {0.0f};
    float (&tdS)[4][2] = tdP;
    // first load K, V into shared memory
    // everything is TILE_SIZE * HEAD_SIZE in row major

    for (int i=0; i< 4;i ++){
        FLOAT4(sK(thread_row_copy + i, thread_col_copy)) = FLOAT4(gK(thread_row_copy + i, thread_col_copy));
        FLOAT4(sV(thread_row_copy + i, thread_col_copy)) = FLOAT4(gV(thread_row_copy + i, thread_col_copy));
    }

    for (int q_tile = blockIdx.y; q_tile < T / TILE_SIZE; q_tile++) {
//         if (thread_id ==0 ){
//
//             printf("blockIdx.y = %d, q_tile = %d, T/TILE_SIZE = %d\n", blockIdx.y, q_tile, T / TILE_SIZE);
//         }

        // load Q, dQ, dO into shared memory
        for (int i=0; i < 4;i ++){
            FLOAT4(sQ(thread_row_copy + i, thread_col_copy)) = FLOAT4(gQ(thread_row_copy + i, thread_col_copy));
            FLOAT4(sQ(thread_row_copy + i, thread_col_copy)) = FLOAT4(gQ(thread_row_copy + i, thread_col_copy));

            FLOAT4(sdQ(thread_row_copy + i, thread_col_copy)) = FLOAT4(gdQ(thread_row_copy + i, thread_col_copy));
            FLOAT4(sdQ(thread_row_copy + i, thread_col_copy)) = FLOAT4(gdQ(thread_row_copy + i, thread_col_copy));

            FLOAT4(sdO(thread_row_copy + i, thread_col_copy)) = FLOAT4(gdO(thread_row_copy + i, thread_col_copy));
            FLOAT4(sdO(thread_row_copy + i, thread_col_copy)) = FLOAT4(gdO(thread_row_copy + i, thread_col_copy));
        }

        __syncthreads();

        // load l, d into registers
        for (int i=0; i< 4;i ++){
            rL[i] = gL(thread_row_copy + i);
            rD[i] = gD(thread_row_copy + i);
        }


        //
        // dV
        //


        // reset tS back to zero
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                tS[i][j] = 0;
            }
        }


        // compute S = Q * K^T
        for (int k_fragment = 0; k_fragment < HEAD_SIZE; k_fragment++) {
            for (int i = 0; i < 4; i++) {
                rQ[i] = sQ(thread_row_32_x_32 + i, k_fragment);
            }
            for (int i = 0; i < 2; i++) {
                rK[i] = sK_T(k_fragment, thread_col_32_x_32 + i);
            }

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 2; j++) {
                    if (q_tile == blockIdx.y  && thread_row_32_x_32 + i < thread_col_32_x_32 + j) {
                        tS[i][j] = -FLT_MAX;
                    } else {
                        tS[i][j] += rQ[i] * rK[j];
                    }
                }
            }
        }

//         if (thread_id == 0) {
//             printf("tS\n");
//             for (int i = 0; i < 4; i++) {
//                 for (int j = 0; j < 2; j++) {
//                     printf("%.2f ", tS[i][j]);
//                 }
//                 printf("\n");
//             }
//         }

        //rescale preatt by 1/sqrt(HS)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                if (tS[i][j] != -FLT_MAX) {
                    tS[i][j] *= 1.0f / sqrtf(HS);
                }
            }
        }

//         if (thread_id == 0) {
//             printf("tS scaled\n");
//             for (int i = 0; i < 4; i++) {
//                 for (int j = 0; j < 2; j++) {
//                     printf("%.2f ", tS[i][j]);
//                 }
//                 printf("\n");
//             }
//         }


        // compute P = exp(Q * K^T - l)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
               tP[i][j] = expf(tS[i][j] - rL[i]);
            }
        }

//         if (thread_id == 0) {
//             printf("tP\n");
//             for (int i = 0; i < 4; i++) {
//                 for (int j = 0; j < 2; j++) {
//                     printf("%.2f ", tP[i][j]);
//                 }
//                 printf("\n");
//             }
//         }

        // store to sP
        //
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
               sP(thread_row_32_x_32 + i, thread_col_32_x_32 + j) = tP[i][j];
            }
        }

        __syncthreads();

        // compute dV = dV + P^T * dO
        for (int k_fragment = 0; k_fragment < TILE_SIZE; k_fragment++) {
            for (int i=0;i<4;i++) {
                rP[i] = sP_T(thread_row_32_x_64 + i, k_fragment);
                rdO[i] = sdO(k_fragment, thread_col_32_x_64 + i);
            }
            for (int i=0; i<4; i++) {
                for (int j=0; j<4; j++) {
                    tdV[i][j] += rP[i] * rdO[j];
                }
            }
        }

//         if (thread_id == 0) {
//             printf("tdV\n");
//             for (int i = 0; i < 4; i++) {
//                 for (int j = 0; j < 4; j++) {
//                     printf("%.2f ", tdV[i][j]);
//                 }
//                 printf("\n");
//             }
//         }


        //
        // dK
        //

        // todo need to scale 1/sqrt(HS) for dK and dQ?
        // also need masking on dP and dS
//         if (thread_id == 0) {
//             printf("tdP init\n");
//             for (int i = 0; i < 4; i++) {
//                 for (int j = 0; j < 2; j++) {
//                     printf("%.2f ", tdP[i][j]);
//                 }
//                 printf("\n");
//             }
//         }
//         if (thread_id == 0) {
//             printf("tdS init\n");
//             for (int i = 0; i < 4; i++) {
//                 for (int j = 0; j < 2; j++) {
//                     printf("%.2f ", tdS[i][j]);
//                 }
//                 printf("\n");
//             }
//         }

//         if (thread_id == 0) {
//             printf("tdK init\n");
//             for (int i = 0; i < 4; i++) {
//                 for (int j = 0; j < 4; j++) {
//                     printf("%.2f ", tdK[i][j]);
//                 }
//                 printf("\n");
//             }
//         }

        // reset tdP back to zero
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                tdP[i][j] = 0;
            }
        }


        // compute dP = dO * V^T
        for (int k_fragment = 0; k_fragment < HEAD_SIZE; k_fragment++) {
            for (int i = 0; i < 4; i++) {
                rdO[i] = sdO(thread_row_32_x_32 + i, k_fragment);

            }
            for (int i = 0; i < 2; i++) {
                rV[i] = sV_T(k_fragment, thread_col_32_x_32 + i);
            }

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 2; j++) {
                    if (q_tile == blockIdx.y  && thread_row_32_x_32 + i < thread_col_32_x_32 + j) {
                        tdP[i][j] = 0;
                    } else {
                        tdP[i][j] += rdO[i] * rV[j];
                    }

                }
            }
        }

//         if (thread_id == 0) {
//             printf("tdP\n");
//             for (int i = 0; i < 4; i++) {
//                 for (int j = 0; j < 2; j++) {
//                     printf("%.2f ", tdP[i][j]);
//                 }
//                 printf("\n");
//             }
//         }


        // compute dS = P \circ (dP - D)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                tdS[i][j] = tP[i][j] * (tdP[i][j] - rD[i]);
            }
        }

//         if (thread_id == 0) {
//             printf("tdS\n");
//             for (int i = 0; i < 4; i++) {
//                 for (int j = 0; j < 2; j++) {
//                     printf("%.2f ", tdS[i][j]);
//                 }
//                 printf("\n");
//             }
//         }

        //store dS to shared memory
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 2; j++) {
               sdS(thread_row_32_x_32 + i, thread_col_32_x_32 + j) = tdS[i][j];
            }
        }

        __syncthreads();


        // compute dK = dS^T * Q
        for (int k_fragment = 0; k_fragment < TILE_SIZE; k_fragment++) {

            for (int i=0;i<4;i++) {
                rdS[i] = sdS_T(thread_row_32_x_64 + i, k_fragment);
                rQ[i] = sQ(k_fragment, thread_col_32_x_64 + i);
            }

            for (int i=0;i<4;i++) {
                for (int j=0; j<4; j++) {
                    tdK[i][j] += rdS[i] * rQ[j];
                }
            }
        }

//         if (thread_id == 0) {
//             printf("tdK\n");
//             for (int i = 0; i < 4; i++) {
//                 for (int j = 0; j < 4; j++) {
//                     printf("%.2f ", tdK[i][j]);
//                 }
//                 printf("\n");
//             }
//         }

        //
        // compute dQ
        //

        // reset tdQ back to zero
        for (int i=0;i<4;i++) {
            for (int j=0; j<4; j++) {
                tdQ[i][j] = 0;
            }
        }

        for (int k_fragment = 0; k_fragment < TILE_SIZE; k_fragment++) {

            for (int i=0;i<4;i++) {
                rdS[i] = sdS(thread_row_32_x_64 + i, k_fragment);
                rK[i] = sK(k_fragment, thread_col_32_x_64 + i);
            }

            for (int i=0;i<4;i++) {
                for (int j=0; j<4; j++) {
                    tdQ[i][j] += rdS[i] * rK[j];
                }
            }
        }

        for (int i=0;i<4;i++) {
            for (int j=0; j<4; j++) {
                tdQ[i][j] *= 1.0f / sqrtf(HS);
            }
        }

        // store dQ

//         if (thread_id == 0) {
//             for (int i = 0; i < 4; i++) {
//                 for (int j = 0; j < 4; j++) {
//                     printf("blockIdx.y = %d, q_tile = %d, i = %d, j=%d, tdQ scaled = %.6f\n", blockIdx.y, q_tile, i, j, tdQ[i][j]);
//                 }
//             }
//         }


//         for (int i=0;i<4;i++) {
//             for (int j=0; j<4; j++) {
//                 atomicAdd(&gdQ(thread_row_copy + i, thread_col_copy + j ), tdQ[i][j]);
//             }
//         }

        for (int i=0;i<4;i++) {
            for (int j=0; j<4; j++) {
                sdQ(thread_row_copy + i, thread_col_copy + j) = tdQ[i][j];
            }
        }
        __syncthreads();

        for (int i=0;i<4;i++) {
            atomicAdd(&gdQ(thread_row_atomic_add + i, thread_col_atomic_add ), sdQ(thread_row_atomic_add + i, thread_col_atomic_add));
            atomicAdd(&gdQ(thread_row_atomic_add + i, thread_col_atomic_add + 32), sdQ(thread_row_atomic_add + i, thread_col_atomic_add + 32));
            atomicAdd(&gdQ(thread_row_atomic_add + i + 4, thread_col_atomic_add ), sdQ(thread_row_atomic_add + i + 4, thread_col_atomic_add));
            atomicAdd(&gdQ(thread_row_atomic_add + i + 4, thread_col_atomic_add + 32), sdQ(thread_row_atomic_add + i + 4, thread_col_atomic_add + 32));
        }

        gQ += qkv_increment;
        gdQ += qkv_increment;
        gdO += o_increment;
        gL += ld_increment;
        gD += ld_increment;
        __syncthreads();
    }

    // rescale dK
    for (int i=0;i<4;i++) {
        for (int j=0; j<4; j++) {
            tdK[i][j] *= 1.0f / sqrtf(HS);
        }
    }

//     if (thread_id == 0) {
//         printf("tdK scaled\n");
//         for (int i = 0; i < 4; i++) {
//             for (int j = 0; j < 4; j++) {
//                 printf("%.2f ", tdK[i][j]);
//             }
//             printf("\n");
//         }
//     }

    // store dK to global memory

//     for (int i=0;i<4;i++) {
//         for (int j=0; j<4; j++) {
//             gdK(thread_row_copy + i , thread_col_copy + j) = tdK[i][j];
//         }
//     }

    for (int i=0;i<4;i++) {
        FLOAT4(gdK(thread_row_copy + i , thread_col_copy)) = FLOAT4(tdK[i][0]);
    }


//     // store dV to global memory
//     for (int i=0;i<4;i++) {
//         for (int j=0; j<4; j++) {
//             gdV(thread_row_copy + i , thread_col_copy + j) = tdV[i][j];
//         }
//     }
    for (int i=0;i<4;i++) {
        FLOAT4(gdV(thread_row_copy + i , thread_col_copy)) = FLOAT4(tdV[i][0]);
    }

}

#undef TILE_SIZE

#define TILE_SIZE 64
// #define HEAD_SIZE 64
//
// #define gQ(i,j) gQ[(i) * 3 * NH * HS + (j)]
// #define gK(i,j) gK[(i) * 3 * NH * HS + (j)]
// #define gV(i,j) gV[(i) * 3 * NH * HS + (j)]
//
// #define gdQ(i,j) gdQ[(i) * 3 * NH * HS + (j)]
// #define gdK(i,j) gdK[(i) * 3 * NH * HS + (j)]
// #define gdV(i,j) gdV[(i) * 3 * NH * HS + (j)]
//
// #define gdO(i,j) gdO[(i) * 1 * NH * HS + (j)]
// #define gL(i) gL[(i) * NH]
// #define gD(i) gD[(i) * NH]
//
// #define sQ(i,j) sQ[(i) * HEAD_SIZE + (j)]
// #define sK(i,j) sK[(i) * HEAD_SIZE + (j)]
// #define sK_T(i,j) sK[(i) + (j) * HEAD_SIZE]
// #define sV(i,j) sV[(i) * HEAD_SIZE + (j)]
// #define sV_T(i,j) sV[(i) + (j) * HEAD_SIZE]
//
// #define sdO(i,j) sdO[(i) * HEAD_SIZE + (j)]
// #define sdQ(i,j) sdQ[(i) * HEAD_SIZE + (j)]
// #define sP(i,j) sP[(i) * TILE_SIZE + (j)]
// #define sP_T(i,j) sP[(i) + (j) * TILE_SIZE]
// #define sdS(i,j) sdS[(i) * TILE_SIZE + (j)]
// #define sdS_T(i,j) sdS[(i) + (j) * TILE_SIZE]
//#define FLOAT4(value) *reinterpret_cast<float4*>(&(value))[0]


__global__ void flash_attention_backward_kernel2(float* dinp, float* inp, float* dout, float* out, float* l, float* d,
                                int B, int T, int NH, int HS) {
    // inp  (B, T, 3, NH, HS)
    // out  (B, T, NH, HS)
    // dout (B, T, NH, HS)
    // l    (B, T, NH)
    // d    (B, T, NH)

    // dinp (B, T, 3, NH, HS)

    // blockDim.x = NH
    // blockDim.y = T
    // blockDim.z = B

    // High level overview:
    // 1. load K, V into shared memory
    // 2. load Q, dO into registers
    // 3. compute S = Q * K^T -> P and dP = dO * V^T -> dS in registers
    // 4. store P, dO in shared memory
    // 5. compute dV = P^T * dO in registers
    // 6. store dS, Q in shared memory
    // 7. compute dK = dS^T * Q and dQ = dS * K in registers
    // Each Q, K, V, Q, P, dS, dO is a 64 x 64 matrix which uses 16KB of shared memory
    // At any one point we only need to store 4 of these tiles in shared memory so we only need 64KB of shared memory,
    // so that this kernel would work for any GPUs with SM >= 7.0


    int thread_id = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;


    int q_global_offset_start = blockIdx.z * T * 3 * NH * HS + 0 * 3 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int k_global_offset_start = blockIdx.z * T * 3 * NH * HS + 0 * 3 * NH * HS + 1 * NH * HS + blockIdx.x * HS;
    int v_global_offset_start = blockIdx.z * T * 3 * NH * HS + 0 * 3 * NH * HS + 2 * NH * HS + blockIdx.x * HS;
    int o_global_offset_start = blockIdx.z * T * 1 * NH * HS + 0 * 1 * NH * HS + 0 * NH * HS + blockIdx.x * HS;

    // offset for the q,k,v of the corresponding head
    int q_global_offset_current = blockIdx.z * T * 3 * NH * HS + blockIdx.y * TILE_SIZE * 3 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int k_global_offset_current = blockIdx.z * T * 3 * NH * HS + blockIdx.y * TILE_SIZE * 3 * NH * HS + 1 * NH * HS + blockIdx.x * HS;
    int v_global_offset_current = blockIdx.z * T * 3 * NH * HS + blockIdx.y * TILE_SIZE * 3 * NH * HS + 2 * NH * HS + blockIdx.x * HS;
    int o_global_offset_current = blockIdx.z * T * 1 * NH * HS + blockIdx.y * TILE_SIZE * 1 * NH * HS + 0 * NH * HS + blockIdx.x * HS;

    // following flashattention 2, we only store (log (L) + m) instead of L, m
    int ld_global_offset_start = blockIdx.z * T * NH + 0 * NH + blockIdx.x;
    int ld_global_offset_current = blockIdx.z * T * NH + blockIdx.y * TILE_SIZE * NH + blockIdx.x;

    int qkv_increment = TILE_SIZE * 3 * NH * HS;
    int o_increment = TILE_SIZE * NH * HS;
    int ld_increment = TILE_SIZE * NH;

    float* gQ = &inp[q_global_offset_current];
    float* gK = &inp[k_global_offset_current];
    float* gV = &inp[v_global_offset_current];
    float* gdO = &dout[o_global_offset_current];
    float* gL = &l[ld_global_offset_current];
    float* gD = &d[ld_global_offset_current];


    // output
    float* gdQ = &dinp[q_global_offset_current];
    float* gdK = &dinp[k_global_offset_current];
    float* gdV = &dinp[v_global_offset_current];

    extern __shared__ float sharedMemory[];

    float* sK = &sharedMemory[0 * 64 * 64];
    float* sV = &sharedMemory[1 * 64 * 64];

    float* sP = &sharedMemory[2 * 64 * 64];
    float* sdS = &sharedMemory[2 * 64 * 64];

    float* sdO = &sharedMemory[3 * 64 * 64];
    float* sQ  = &sharedMemory[3 * 64 * 64];
    float* sdQ = &sharedMemory[3 * 64 * 64];


    // offset for loading from global to shared and register tiling
    int thread_row = warp_id * 8 + (lane_id / 16) * 4;
    int thread_col = (lane_id % 16) * 4;


    // offset for atomic add for dQ
    int thread_row_atomic_add = warp_id * 8;
    int thread_col_atomic_add = lane_id;


    unsigned mask = (lane_id < 16) ? 0xFFFF : 0xFFFF0000; // Mask for the two halves
    //int lane_id_to_read_from = (lane_id < 16) ? 0 : 16; // Lane to read from

    float rL[4];
    float rD[4];

    float rQ[4];
    float rK[4];
    float rV[4];
    float rdO[4];
    float rP[4];
    float rdS[4];

    float tQ[4][4];
    float tdO[4][4];

    float tdQ[4][4] = {0.0f};
    float tdK[4][4] = {0.0f};
    float tdV[4][4] = {0.0f};

    float tS[4][4] = {0.0f};
    float (&tP)[4][4] = tS;
    float tdP[4][4] = {0.0f};
    float (&tdS)[4][4] = tdP;
    // first load K, V into shared memory
    // everything is TILE_SIZE * HEAD_SIZE in row major

    for (int i=0; i < 4;i ++){
        FLOAT4(sK(thread_row + i, thread_col)) = FLOAT4(gK(thread_row + i, thread_col));
        FLOAT4(sV(thread_row + i, thread_col)) = FLOAT4(gV(thread_row + i, thread_col));
    }

    __syncthreads();

    for (int q_tile = blockIdx.y; q_tile < T / TILE_SIZE; q_tile++) {

        // load Q, dO into registers
        for (int i=0; i < 4;i ++){
            FLOAT4(tQ[i][0]) = FLOAT4(gQ(thread_row + i, thread_col));
            FLOAT4(tdO[i][0]) = FLOAT4(gdO(thread_row + i, thread_col));
        }

        if (blockIdx.y==0 && thread_id ==0 and q_tile == 0){
            printf("kernel 2");
            for (int i = 0; i < 64; i++) {
                for (int j=0;j<64;j++) {
                    printf("%f ", sQ(i,j));
                }
                print("\n");
            }
            print("==========");
        }

        // load l, d into registers
        for (int i=0; i< 4;i ++){
            rL[i] = gL(thread_row + i);
            rD[i] = gD(thread_row + i);
        }


        //
        // compute S and P
        //

        // reset tS back to zero
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                tS[i][j] = 0;
            }
        }

        // compute S = Q * K^T
        for (int k_fragment_outer = 0; k_fragment_outer < 16; k_fragment_outer++) {
            for (int k_fragment_inner = 0; k_fragment_inner < 4; k_fragment_inner++) {
                // position is k_fragment_outer * 4 + k_fragment_inner
                int k_fragment = k_fragment_outer * 4 + k_fragment_inner;
                for (int i = 0; i < 4; i++) {
                    rQ[i] = __shfl_sync(mask, tQ[i][k_fragment_inner], (lane_id /16) * 16  + k_fragment_outer);
                    rK[i] = sK_T(k_fragment, thread_col + i);
                }

                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        if (q_tile == blockIdx.y  && thread_row + i < thread_col + j) {
                            tS[i][j] = -FLT_MAX;
                        } else {
                            tS[i][j] += rQ[i] * rK[j];
                        }
                    }
                }
            }
        }

//         if (blockIdx.y == 0 && thread_id == 16) {
//             printf("kernel 2, tS, q_tile = %d\n", q_tile);
//             for (int i=0;i<4;i++) {
//                 for (int j=0;j<4;j++) {
//                     printf("%f ", tS[i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("==========\n");
//         }

        //rescale S by 1/sqrt(HS)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (tS[i][j] != -FLT_MAX) {
                    tS[i][j] *= 1.0f / sqrtf(HS);
                }
            }
        }

        // compute P = exp(Q * K^T - l)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
               tP[i][j] = expf(tS[i][j] - rL[i]);
            }
        }

//         if (blockIdx.y == 0 && thread_id == 16) {
//             printf("kernel 2, tP, q_tile = %d\n", q_tile);
//             for (int i=0;i<4;i++) {
//                 for (int j=0;j<4;j++) {
//                     printf("%f ", tP[i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("==========\n");
//         }

        //
        // compute dP and dS
        //

        // reset tdP back to zero
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                tdP[i][j] = 0;
            }
        }

        // compute dP = dO * V^T
        for (int k_fragment_outer = 0; k_fragment_outer < 16; k_fragment_outer++) {
            for (int k_fragment_inner = 0; k_fragment_inner < 4; k_fragment_inner++) {
                // position is k_fragment_outer * 4 + k_fragment_inner
                int k_fragment = k_fragment_outer * 4 + k_fragment_inner;
                for (int i = 0; i < 4; i++) {
                    rdO[i] = __shfl_sync(mask, tdO[i][k_fragment_inner], (lane_id /16) * 16  + k_fragment_outer);
                    rV[i] = sV_T(k_fragment, thread_col + i);
                }

                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        if (q_tile == blockIdx.y  && thread_row + i < thread_col + j) {
                            tdP[i][j] = 0;
                        } else {
                            tdP[i][j] += rdO[i] * rV[j];
                        }
                    }
                }
            }
        }

//         if (blockIdx.y == 0 && thread_id == 16) {
//             printf("kernel 2, tdP, q_tile = %d\n", q_tile);
//             for (int i=0;i<4;i++) {
//                 for (int j=0;j<4;j++) {
//                     printf("%f ", tdP[i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("==========\n");
//         }

        // compute dS = P \circ (dP - D)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                tdS[i][j] = tP[i][j] * (tdP[i][j] - rD[i]);
            }
        }

//         if (blockIdx.y == 0 && thread_id == 16) {
//             printf("kernel 2, tdS, q_tile = %d\n", q_tile);
//             for (int i=0;i<4;i++) {
//                 for (int j=0;j<4;j++) {
//                     printf("%f ", tdS[i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("==========\n");
//         }

        //
        //  compute dV
        //

        // store P to shared memory
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
               sP(thread_row + i, thread_col + j) = tP[i][j];
            }
        }

        // store dO in shared memory
        for (int i=0; i < 4;i ++){
            FLOAT4(sdO(thread_row+i, thread_col)) = FLOAT4(tdO[i][0]);
        }

        __syncthreads();

        // compute dV = P^T * dO
        for (int k_fragment = 0; k_fragment < TILE_SIZE; k_fragment++) {
            for (int i = 0; i < 4;i++) {
                rP[i] = sP_T(thread_row + i, k_fragment);
                rdO[i] = sdO(k_fragment, thread_col + i);
            }

//             if (blockIdx.y == 0 && thread_id == 0 ) {
//                 printf("kernel 2, k_fragment = %d\n", k_fragment);
//                 printf("rP = ");
//                 for (int i = 0; i < 4; i++) {
//                     printf("%f ", rP[i]);
//                 }
//                 printf("\n");
//
//                 printf("rdO = ");
//                 for (int i = 0; i < 4; i++) {
//                     printf("%f ", rdO[i]);
//                 }
//                 printf("\n");
//
//             }

            for (int i=0; i<4; i++) {
                for (int j=0; j<4; j++) {
                    tdV[i][j] += rP[i] * rdO[j];
                }
            }
        }

//         if (blockIdx.y == 0 && thread_id == 0) {
//             printf("kernel 2, tdV, q_tile = %d\n", q_tile);
//             for (int i=0;i<4;i++) {
//                 for (int j=0;j<4;j++) {
//                     printf("%f ", tdV[i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("==========\n");
//         }


        __syncthreads();

        //
        // dK
        //

        //store dS to shared memory
        for (int i = 0; i < 4; i++) {

           FLOAT4(sdS(thread_row + i, thread_col)) = FLOAT4(tdS[i][0]);
        }

        //store Q to shared memory
        for (int i =0; i<4; i++) {
            FLOAT4(sQ(thread_row + i, thread_col)) = FLOAT4(tQ[i][0]);
        }

        __syncthreads();

        // compute dK = dS^T * Q
        for (int k_fragment = 0; k_fragment < TILE_SIZE; k_fragment++) {

            for (int i = 0; i < 4; i++) {
                rdS[i] = sdS_T(thread_row + i, k_fragment);
                rQ[i] = sQ(k_fragment, thread_col + i);
            }
//             if (blockIdx.y == 0 && thread_id == 0 ) {
//                 printf("kernel 2, k_fragment = %d\n", k_fragment);
//                 printf("rdS = ");
//                 for (int i = 0; i < 4; i++) {
//                     printf("%f ", rdS[i]);
//                 }
//                 printf("\n");
//
//                 printf("rQ = ");
//                 for (int i = 0; i < 4; i++) {
//                     printf("%f ", rQ[i]);
//                 }
//                 printf("\n");
//             }
            for (int i = 0; i < 4; i++) {
                for (int j=0; j<4; j++) {
                    tdK[i][j] += rdS[i] * rQ[j];
                }
            }
        }

//         if (blockIdx.y == 0 && thread_id == 0) {
//             printf("kernel 2, tdK, q_tile = %d\n", q_tile);
//             for (int i=0;i<4;i++) {
//                 for (int j=0;j<4;j++) {
//                     printf("%f ", tdK[i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("==========\n");
//         }
        //
        // compute dQ
        //

        // reset tdQ back to zero
        for (int i=0;i<4;i++) {
            for (int j=0; j<4; j++) {
                tdQ[i][j] = 0;
            }
        }

        for (int k_fragment = 0; k_fragment < TILE_SIZE; k_fragment++) {

            for (int i=0;i<4;i++) {
                rdS[i] = sdS(thread_row + i, k_fragment);
                rK[i] = sK(k_fragment, thread_col + i);
            }

            for (int i=0;i<4;i++) {
                for (int j=0; j<4; j++) {
                    tdQ[i][j] += rdS[i] * rK[j];
                }
            }
        }

        for (int i=0;i<4;i++) {
            for (int j=0; j<4; j++) {
                tdQ[i][j] *= 1.0f / sqrtf(HS);
            }
        }

        // store dQ
        for (int i=0;i<4;i++) {
            for (int j=0; j<4; j++) {
                sdQ(thread_row + i, thread_col + j) = tdQ[i][j];
            }
        }
        __syncthreads();

        for (int i=0;i<4;i++) {
            atomicAdd(&gdQ(thread_row_atomic_add + i, thread_col_atomic_add ), sdQ(thread_row_atomic_add + i, thread_col_atomic_add));
            atomicAdd(&gdQ(thread_row_atomic_add + i, thread_col_atomic_add + 32), sdQ(thread_row_atomic_add + i, thread_col_atomic_add + 32));
            atomicAdd(&gdQ(thread_row_atomic_add + i + 4, thread_col_atomic_add ), sdQ(thread_row_atomic_add + i + 4, thread_col_atomic_add));
            atomicAdd(&gdQ(thread_row_atomic_add + i + 4, thread_col_atomic_add + 32), sdQ(thread_row_atomic_add + i + 4, thread_col_atomic_add + 32));
        }

        gQ += qkv_increment;
        gdQ += qkv_increment;
        gdO += o_increment;
        gL += ld_increment;
        gD += ld_increment;
        __syncthreads();
    }

    // rescale dK
    for (int i=0;i<4;i++) {
        for (int j=0; j<4; j++) {
            tdK[i][j] *= 1.0f / sqrtf(HS);
        }
    }


    // store dK to global memory

    for (int i=0;i<4;i++) {
        FLOAT4(gdK(thread_row + i , thread_col)) = FLOAT4(tdK[i][0]);
    }


//     // store dV to global memory
    for (int i=0;i<4;i++) {
        FLOAT4(gdV(thread_row + i , thread_col)) = FLOAT4(tdV[i][0]);
    }
}


#undef TILE_SIZE
#undef HEAD_SIZE
#undef gQ
#undef gK
#undef gV

#undef gdQ
#undef gdK
#undef gdV

#undef gdO
#undef gL
#undef gD

#undef sQ
#undef sK
#undef sK_T
#undef sV
#undef sV_T

#undef sdO
#undef sdQ
#undef sP
#undef sP_T
#undef sdS
#undef sdS_T


#define Q_TILE_SIZE 64
#define KV_TILE_SIZE 128
#define HEAD_SIZE 64
//
#define gQ(i,j) gQ[(i) * 3 * NH * HS + (j)]
#define gK(i,j) gK[(i) * 3 * NH * HS + (j)]
#define gV(i,j) gV[(i) * 3 * NH * HS + (j)]
//
#define gdQ(i,j) gdQ[(i) * 3 * NH * HS + (j)]
#define gdK(i,j) gdK[(i) * 3 * NH * HS + (j)]
#define gdV(i,j) gdV[(i) * 3 * NH * HS + (j)]
//
#define gdO(i,j) gdO[(i) * 1 * NH * HS + (j)]
#define gL(i) gL[(i) * NH]
#define gD(i) gD[(i) * NH]
//
#define sQ(i,j) sQ[(i) * HEAD_SIZE + (j)]
#define sK(i,j) sK[(i) * HEAD_SIZE + (j)]
#define sK_T(i,j) sK[(i) + (j) * HEAD_SIZE]
// #define sV(i,j) sV[(i) * HEAD_SIZE + (j)]
// #define sV_T(i,j) sV[(i) + (j) * HEAD_SIZE]
//
#define sdO(i,j) sdO[(i) * HEAD_SIZE + (j)]
#define sdQ(i,j) sdQ[(i) * HEAD_SIZE + (j)]
// #define sP(i,j) sP[(i) * TILE_SIZE + (j)]
// #define sP_T(i,j) sP[(i) + (j) * TILE_SIZE]
#define sdS(i,j) sdS[(i) * KV_TILE_SIZE + (j)]
#define sdS_T(i,j) sdS[(i) + (j) * KV_TILE_SIZE]
//#define FLOAT4(value) *reinterpret_cast<float4*>(&(value))[0]


__global__ __launch_bounds__(256)
void flash_attention_backward_kernel3(float* dinp, float* inp, float* dout, float* out, float* l, float* d,
                                int B, int T, int NH, int HS) {
    // inp  (B, T, 3, NH, HS)
    // out  (B, T, NH, HS)
    // dout (B, T, NH, HS)
    // l    (B, T, NH)
    // d    (B, T, NH)

    // dinp (B, T, 3, NH, HS)

    // blockDim.x = NH
    // blockDim.y = T
    // blockDim.z = B

    // K in shared memory, V in register

    int thread_id = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;


    int q_global_offset_start = blockIdx.z * T * 3 * NH * HS + 0 * 3 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int k_global_offset_start = blockIdx.z * T * 3 * NH * HS + 0 * 3 * NH * HS + 1 * NH * HS + blockIdx.x * HS;
    int v_global_offset_start = blockIdx.z * T * 3 * NH * HS + 0 * 3 * NH * HS + 2 * NH * HS + blockIdx.x * HS;
    int o_global_offset_start = blockIdx.z * T * 1 * NH * HS + 0 * 1 * NH * HS + 0 * NH * HS + blockIdx.x * HS;

    // offset for the q,k,v of the corresponding head
    int q_global_offset_current = blockIdx.z * T * 3 * NH * HS + blockIdx.y * 2 * Q_TILE_SIZE * 3 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int k_global_offset_current = blockIdx.z * T * 3 * NH * HS + blockIdx.y * KV_TILE_SIZE * 3 * NH * HS + 1 * NH * HS + blockIdx.x * HS;
    int v_global_offset_current = blockIdx.z * T * 3 * NH * HS + blockIdx.y * KV_TILE_SIZE * 3 * NH * HS + 2 * NH * HS + blockIdx.x * HS;
    int o_global_offset_current = blockIdx.z * T * 1 * NH * HS + blockIdx.y * 2 * Q_TILE_SIZE * 1 * NH * HS + 0 * NH * HS + blockIdx.x * HS;

    // following flashattention 2, we only store (log (L) + m) instead of L, m
    int ld_global_offset_start = blockIdx.z * T * NH + 0 * NH + blockIdx.x;
    int ld_global_offset_current = blockIdx.z * T * NH + blockIdx.y * 2 * Q_TILE_SIZE * NH + blockIdx.x;

    int q_increment = Q_TILE_SIZE * 3 * NH * HS;
    int o_increment = Q_TILE_SIZE * NH * HS;
    int ld_increment = Q_TILE_SIZE * NH;

    float* gQ = &inp[q_global_offset_current];
    float* gK = &inp[k_global_offset_current];
    float* gV = &inp[v_global_offset_current];
    float* gdO = &dout[o_global_offset_current];
    float* gL = &l[ld_global_offset_current];
    float* gD = &d[ld_global_offset_current];


    // output
    float* gdQ = &dinp[q_global_offset_current];
    float* gdK = &dinp[k_global_offset_current];
    float* gdV = &dinp[v_global_offset_current];

    extern __shared__ float sharedMemory[];

    float* sQ = &sharedMemory[0];
    float* sdO = sQ + Q_TILE_SIZE * Q_TILE_SIZE;
    float* sK = sdO + Q_TILE_SIZE * Q_TILE_SIZE;
    float* sdS = sQ;
    float* sdQ = sQ;

    // offset for loading from global to shared
    int thread_row_copy = warp_id * 8 + (lane_id / 16) * 4;
    int thread_col_copy = (lane_id % 16) * 4;

    // offset for register tiling for dK, dV
    int thread_row_128_x_64 = warp_id * 16 + (lane_id / 16) * 4;
    int thread_col_128_x_64 = (lane_id % 16) * 4;

    int thread_row_64_x_128 = thread_col_128_x_64;
    int thread_col_64_x_128 = thread_row_128_x_64;

    int thread_row_64_x_64 = warp_id * 8 + (lane_id / 16) * 4;
    int thread_col_64_x_64 = (lane_id % 16) * 4;


    // offset for register tiling for S and dP
    int thread_row_reg_tiling = thread_col_copy;
    int thread_col_reg_tiling = thread_row_copy;



    // offset for atomic add for dQ
    int thread_row_atomic_add = warp_id * 8;
    int thread_col_atomic_add = lane_id;

    unsigned mask = (lane_id < 16) ? 0xFFFF : 0xFFFF0000; // Mask for the two halves

    float rL[4];
    float rD[4];
//
    float rQ[4];
    float rK[8];
    float rV[8];
    float rdO[4];
    float rP[8];
    float rdS[8];
//
//     float tQ[4][4];
//     float tdO[4][4];
//
    float tV[8][4];
    float tdQ[4][4] = {0.0f};
    float tdK[8][4] = {0.0f};
    float tdV[8][4] = {0.0f};
//
    float tS[4][8] = {0.0f};
    float (&tP)[4][8] = tS;
    float tdP[4][8] = {0.0f};
    float (&tdS)[4][8] = tdP;
    // first load K into shared memory and V into registers
    // everything is TILE_SIZE * HEAD_SIZE in row major

    for (int i=0; i < 4;i ++){
        FLOAT4(sK(thread_row_128_x_64 + i, thread_col_128_x_64)) = FLOAT4(gK(thread_row_128_x_64 + i, thread_col_128_x_64));
        FLOAT4(sK(thread_row_128_x_64 + 8 + i,  thread_col_128_x_64)) = FLOAT4(gK(thread_row_128_x_64 + 8 + i, thread_col_128_x_64));
        FLOAT4(tV[i][0]) = FLOAT4(gV(thread_row_128_x_64 + i, thread_col_128_x_64));
        FLOAT4(tV[i+4][0]) = FLOAT4(gV(thread_row_128_x_64 + 8 + i, thread_col_128_x_64));
    }

    __syncthreads();

    for (int q_tile = 2 * blockIdx.y; q_tile < T / Q_TILE_SIZE; q_tile++) {

        // load Q, dO into shared memory
        for (int i=0; i < 4;i ++){
            FLOAT4(sQ(thread_row_64_x_64 + i, thread_col_64_x_64)) = FLOAT4(gQ(thread_row_64_x_64 + i, thread_col_64_x_64));
            FLOAT4(sdO(thread_row_64_x_64 + i, thread_col_64_x_64)) = FLOAT4(gdO(thread_row_64_x_64 + i, thread_col_64_x_64));
        }

        if (blockIdx.y==0 && thread_id ==0 and q_tile == 0){
            printf("kernel 3");
            for (int i = 0; i < 64; i++) {
                for (int j=0;j<64;j++) {
                    printf("%f ", sQ(i,j));
                }
                print("\n");
            }
            print("==========");
        }



        // load l, d into registers
        // wrong row, i think it should be thread_row_64_x_128?
        // original was thread_row_64_x_64
        // check rD, might still be wrong
        for (int i=0; i< 4;i ++){
            rL[i] = gL(thread_row_64_x_128 + i);
            rD[i] = gD(thread_row_64_x_128 + i);
        }

        __syncthreads();

        //
        // compute S and P
        //

        // reset tS back to zero
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                tS[i][j] = 0;
            }
        }

        // compute S = Q * K^T
        for (int k_fragment = 0; k_fragment < HEAD_SIZE; k_fragment++) {
            for (int i = 0; i < 4; i++) {
                rQ[i] = sQ(thread_row_64_x_128 + i, k_fragment);
                rK[i] = sK_T(k_fragment, thread_col_64_x_128 + i);
                rK[i+4] = sK_T(k_fragment, thread_col_64_x_128 + 8 + i);
            }

            //tS is 4 x 8
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (q_tile * Q_TILE_SIZE + thread_row_64_x_128 + i < blockIdx.y * KV_TILE_SIZE + thread_col_64_x_128 + j) {
                        tS[i][j] = -FLT_MAX;
                    } else {
                        tS[i][j] += rQ[i] * rK[j];
                    }

                    if (q_tile * Q_TILE_SIZE + thread_row_64_x_128 + i < blockIdx.y * KV_TILE_SIZE + thread_col_64_x_128 + 8 + j) {
                        tS[i][j + 4] = -FLT_MAX;
                    } else {
                        tS[i][j + 4] += rQ[i] * rK[j + 4];
                    }
                }
            }
        }

//         if (blockIdx.y == 0 && thread_id == 1) {
//             printf("kernel 3, tS, q_tile = %d\n", q_tile);
//             for (int i=0;i<4;i++) {
//                 for (int j=0;j<4;j++) {
//                     printf("%f ", tS[i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("==========\n");
//         }

        //rescale S by 1/sqrt(HS)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                if (tS[i][j] != -FLT_MAX) {
                    tS[i][j] *= 1.0f / sqrtf(HS);
                }
            }
        }

        // compute P = exp(Q * K^T - l)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
               tP[i][j] = expf(tS[i][j] - rL[i]);
            }
        }

//         if (blockIdx.y == 0 && thread_id == 1) {
//             printf("kernel 3, tP, q_tile = %d\n", q_tile);
//             for (int i=0;i<4;i++) {
//                 for (int j=0;j<4;j++) {
//                     printf("%f ", tP[i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("==========\n");
//         }


        //
        // compute dP and dS
        //

        // reset tdP back to zero
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                tdP[i][j] = 0;
            }
        }

        // compute dP = dO * V^T
        for (int k_fragment_outer = 0; k_fragment_outer < 16; k_fragment_outer++) {
            for (int k_fragment_inner = 0; k_fragment_inner < 4; k_fragment_inner++) {
                // position is k_fragment_outer * 4 + k_fragment_inner
                int k_fragment = k_fragment_outer * 4 + k_fragment_inner;
                for (int i = 0; i < 4; i++) {
                    rdO[i] = sdO(thread_row_64_x_128 + i, k_fragment);
                    rV[i] = __shfl_sync(mask, tV[i][k_fragment_inner], (lane_id / 16) * 16  + k_fragment_outer);
                    rV[i+4] = __shfl_sync(mask, tV[i+4][k_fragment_inner], (lane_id / 16) * 16  + k_fragment_outer);
                }

                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        if (q_tile * Q_TILE_SIZE + thread_row_64_x_128 + i < blockIdx.y * KV_TILE_SIZE + thread_col_64_x_128 + j) {
                            tdP[i][j] = 0;
                        } else {
                            tdP[i][j] += rdO[i] * rV[j];
                        }
                    }
                    for (int j = 0; j < 4; j++) {
                        if (q_tile * Q_TILE_SIZE + thread_row_64_x_128 + i < blockIdx.y * KV_TILE_SIZE + thread_col_64_x_128 + 8 + j) {
                            tdP[i][j+4] = 0;
                        } else {
                            tdP[i][j+4] += rdO[i] * rV[j+4];
                        }
                    }

                }



            }
        }

//         if (blockIdx.y == 0 && thread_id == 1) {
//             printf("kernel 3, tdP, q_tile = %d\n", q_tile);
//             for (int i=0;i<4;i++) {
//                 for (int j=0;j<4;j++) {
//                     printf("%f ", tdP[i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("==========\n");
//         }

        // compute dS = P \circ (dP - D)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                tdS[i][j] = tP[i][j] * (tdP[i][j] - rD[i]);
            }
        }

//         if (blockIdx.y == 0 && thread_id == 1) {
//             printf("kernel 3, tdS, q_tile = %d\n", q_tile);
//             for (int i=0;i<4;i++) {
//                 for (int j=0;j<4;j++) {
//                     printf("%f ", tdS[i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("==========\n");
//         }


        //
        //  compute dV
        //

        // compute dV = P^T * dO
        for (int k_fragment_outer = 0; k_fragment_outer < 16; k_fragment_outer++) {
            for (int k_fragment_inner = 0; k_fragment_inner < 4; k_fragment_inner++) {
                // position is k_fragment_outer * 4 + k_fragment_inner
                int k_fragment = k_fragment_outer * 4 + k_fragment_inner;
                for (int i = 0; i < 4; i++) {
                    rP[i] = __shfl_sync(mask, tP[k_fragment_inner][i], (lane_id / 16) * 16  + k_fragment_outer);
                    rP[i+4] = __shfl_sync(mask, tP[k_fragment_inner][i], (lane_id / 16) * 16  + k_fragment_outer);
                    rdO[i] = sdO(k_fragment, thread_col_128_x_64 + i);
                }

//                 if (blockIdx.y == 0 && thread_id == 0 ) {
//                     printf("kernel 3, k_fragment = %d\n", k_fragment);
//                     printf("rP = ");
//                     for (int i = 0; i < 4; i++) {
//                         printf("%f ", rP[i]);
//                     }
//                     printf("\n");
//
//                     printf("rdO = ");
//                     for (int i = 0; i < 4; i++) {
//                         printf("%f ", rdO[i]);
//                     }
//                     printf("\n");
//                 }

                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 4; j++) {
                        tdV[i][j] += rP[i] * rdO[j];
                    }
                }
            }
        }

//         if (blockIdx.y == 0 && thread_id == 0) {
//             printf("kernel 3, tdV, q_tile = %d\n", q_tile);
//             for (int i=0;i<4;i++) {
//                 for (int j=0;j<4;j++) {
//                     printf("%f ", tdV[i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("==========\n");
//         }


        //
        // dK
        //

        // compute dK = dS^T * Q
        for (int k_fragment_outer = 0; k_fragment_outer < 16; k_fragment_outer++) {
            for (int k_fragment_inner = 0; k_fragment_inner < 4; k_fragment_inner++) {
                // position is k_fragment_outer * 4 + k_fragment_inner
                int k_fragment = k_fragment_outer * 4 + k_fragment_inner;
                for (int i = 0; i < 4; i++) {
                    rdS[i] = __shfl_sync(mask, tdS[k_fragment_inner][i], (lane_id / 16) * 16  + k_fragment_outer);
                    rdS[i+4] = __shfl_sync(mask, tdS[k_fragment_inner][i], (lane_id / 16) * 16  + k_fragment_outer);
                    rQ[i] = sQ(k_fragment, thread_col_128_x_64 + i);
                }
//                 if (blockIdx.y == 0 && thread_id == 0 ) {
//                     printf("kernel 3, k_fragment = %d\n", k_fragment);
//                     printf("rdS = ");
//                     for (int i = 0; i < 4; i++) {
//                         printf("%f ", rdS[i]);
//                     }
//                     printf("\n");
//
//                     printf("rQ = ");
//                     for (int i = 0; i < 4; i++) {
//                         printf("%f ", rQ[i]);
//                     }
//                     printf("\n");
//                 }
                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 4; j++) {
                        tdK[i][j] += rdS[i] * rQ[j];
                    }
                }
            }
        }

//         if (blockIdx.y == 0 && thread_id == 0) {
//             printf("kernel 3, tdK, q_tile = %d\n", q_tile);
//             for (int i=0;i<4;i++) {
//                 for (int j=0;j<4;j++) {
//                     printf("%f ", tdK[i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("==========\n");
//         }

        __syncthreads();

        //
        // compute dQ
        //

        // reset tdQ back to zero
        for (int i=0;i<4;i++) {
            for (int j=0; j<4; j++) {
                tdQ[i][j] = 0;
            }
        }

        // store dS to shared memory
        for (int i = 0; i< 4; i++) {
            for (int j=0; j < 4;j++) {
                sdS(thread_row_64_x_128 + i, thread_col_64_x_128 + j) = tdS[i][j];
                sdS(thread_row_64_x_128 + i, thread_col_64_x_128 + j + 8) = tdS[i][j + 4];
            }
        }

        __syncthreads();

        for (int k_fragment = 0; k_fragment < KV_TILE_SIZE; k_fragment++) {

            for (int i=0;i<4;i++) {
                rdS[i] = sdS(thread_row_64_x_64 + i, k_fragment);
                rK[i] = sK(k_fragment, thread_col_64_x_64 + i);
            }

            for (int i=0;i<4;i++) {
                for (int j=0; j<4; j++) {
                    tdQ[i][j] += rdS[i] * rK[j];
                }
            }
        }

        for (int i=0;i<4;i++) {
            for (int j=0; j<4; j++) {
                tdQ[i][j] *= 1.0f / sqrtf(HS);
            }
        }

        __syncthreads();

        // store dQ
        for (int i=0;i<4;i++) {
            for (int j=0; j<4; j++) {
                sdQ(thread_row_64_x_64 + i, thread_col_64_x_64 + j) = tdQ[i][j];
            }
        }
        __syncthreads();

        for (int i=0;i<8;i++) {
            atomicAdd(&gdQ(thread_row_atomic_add + i, thread_col_atomic_add ), sdQ(thread_row_atomic_add + i, thread_col_atomic_add));
            atomicAdd(&gdQ(thread_row_atomic_add + i, thread_col_atomic_add + 32), sdQ(thread_row_atomic_add + i, thread_col_atomic_add + 32));
        }

        gQ += q_increment;
        gdQ += q_increment;
        gdO += o_increment;
        gL += ld_increment;
        gD += ld_increment;
        __syncthreads();
    }

    // rescale dK
    for (int i=0;i<8;i++) {
        for (int j=0; j<4; j++) {
            tdK[i][j] *= 1.0f / sqrtf(HS);
        }
    }


    // store dK to global memory

    for (int i=0;i<4;i++) {
        FLOAT4(gdK(thread_row_128_x_64 + i ,thread_col_128_x_64)) = FLOAT4(tdK[i][0]);
        FLOAT4(gdK(thread_row_128_x_64 + 8 + i ,thread_col_128_x_64)) = FLOAT4(tdK[i+4][0]);
    }


//     // store dV to global memory
    for (int i=0;i<4;i++) {
        FLOAT4(gdV(thread_row_128_x_64 + i ,thread_col_128_x_64)) = FLOAT4(tdV[i][0]);
        FLOAT4(gdV(thread_row_128_x_64 + 8 + i ,thread_col_128_x_64)) = FLOAT4(tdV[i+4][0]);
    }
}





__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

__global__ void permute_kernel_backward(float* dinp,
                                        const float* dq, const float* dk, const float* dv,
                                        int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        dinp[inp_idx] += dq[idx];
        dinp[inp_idx + NH * d] += dk[idx];
        dinp[inp_idx + 2 * (NH * d)] += dv[idx];
    }
}

__global__ void unpermute_kernel(const float* inp, float *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}

__global__ void unpermute_kernel_backward(float* dinp, const float *dout, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        dinp[idx] += dout[other_idx];
    }
}

__device__ float& vec_at(float4& vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}

__device__ float vec_at(const float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

__global__ void softmax_forward_kernel5(float* out, float inv_temperature, const float* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4  == 0);
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const float* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = warp.thread_rank(); i < pos_by_4; i += warp.size()) {
        float4 v = x_vec[i];
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, vec_at(v, k));
        }
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (vec_at(v, k) - maxval));
        }
    }

    if(4*pos_by_4 + warp.thread_rank() <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, x[4*pos_by_4 + warp.thread_rank()]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * (x[4*pos_by_4 + warp.thread_rank()] - maxval));
    }

    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    sumval *= expf(inv_temperature * (maxval - global_maxval));

    float sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = warp.thread_rank(); i <= own_pos; i += warp.size()) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = expf(inv_temperature * (__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, ev * norm);
    }
}

// naive kernel to backward through an autoregressive softmax, just to get correctness
__global__ void softmax_autoregressive_backward_kernel1(float* dpreatt, const float* datt, const float* att,
                                                     int B, int T, int C, int NH) {
    // dpreatt, datt, att are all (B, NH, T, T)
    int t3 = blockIdx.x * blockDim.x + threadIdx.x;
    if (t3 < T) {
        int hs = C / NH; // head size
        float scale = 1.0f / sqrtf(hs);
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < NH; h++) {
                for (int t = t3; t < T; t++) {
                    const float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                    const float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                    float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                    float accum = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        accum +=  scale * local_derivative * datt_bth[t2];
                    }
                    dpreatt_bth[t3] = accum;
                }
            }
        }
    }
}

// parallelize across t,b,h
__global__ void softmax_autoregressive_backward_kernel2(float* dpreatt, const float* datt, const float* att,
                                                     int B, int T, int C, int NH) {
    int t3 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y * T * T;
    if (t3 >= T) { return; }

    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    for (int t = t3; t < T; t++) {
        float result = 0.0;
        const float* att_bth = att + idx + t*T;
        const float* datt_bth = datt + idx + t*T;
        float* dpreatt_bth = dpreatt + idx + t*T;

        for (int t2 = 0; t2 <= t; t2++) {
            float indicator = t2 == t3 ? 1.0f : 0.0f;
            float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
            result += scale * local_derivative * datt_bth[t2];
        }

        dpreatt_bth[t3] = result;
    }
}

// parallelize across t,b,h
__global__ void softmax_autoregressive_backward_kernel3(float* dpreatt, const float* datt, const float* att,
                                                     int B, int T, int C, int NH) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int t3 = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();

    int idx = blockIdx.y * T * T;
    if (t3 >= T) { return; }

    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    for (int t = t3; t < T; t++) {
        float result = 0.0;
        const float* att_bth = att + idx + t*T;
        const float* datt_bth = datt + idx + t*T;
        float* dpreatt_bth = dpreatt + idx + t*T;
        const float att_at_t3 = att_bth[t3];

        for (int t2 = warp.thread_rank(); t2 <= t; t2 += warp.size()) {
            float indicator = t2 == t3 ? 1.0f : 0.0f;
            float local_derivative = att_bth[t2] * (indicator - att_at_t3);
            result += local_derivative * datt_bth[t2];
        }

        result = cg::reduce(warp, result, cg::plus<float>());
        if(warp.thread_rank() == 0) {
            dpreatt_bth[t3] = scale * result;
        }
    }
}
__global__ void softmax_autoregressive_backward_kernel4(float* __restrict__ dpreatt, const float* __restrict__ datt,
                                                        const float* __restrict__ att,
                                                        int B, int T, int C, int NH) {
    constexpr int UNROLL = 8;
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int t3 = UNROLL * (blockIdx.x * warp.meta_group_size() + warp.meta_group_rank());

    int idx = blockIdx.y * T * T;
    if (t3 >= T) { return; }

    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);

    // the innermost loop combines different values of t2 with different values of t.
    // by handling [t3, t3 + UNROLL) in one thread, we get much better memory reuse:
    // any t3/t-dependent value can be loaded once before the t2 loop.
    // within the t2 loop, we can combine each loaded value with each of the UNROLL
    // pre-loaded values, thus cutting memory ready by a factor of ~UNROLL.

    // one iteration of this loop has to handle the cases
    // this may lead to some invalid indices; therefore, we have several
    // early-outs in the iteration over k below.
    for (int t = t3; t < T; t++) {
        float result[UNROLL] = {};
        const float* att_bth = att + idx + t * T;
        const float* datt_bth = datt + idx + t * T;
        float* dpreatt_bth = dpreatt + idx + t * T;

        float att_at_t3[UNROLL];
        for(int k = 0; k < UNROLL; ++k) {
            if (t < t3 + k) continue;
            att_at_t3[k] = att_bth[t3 + k];
        }

        for (int t2 = warp.thread_rank(); t2 <= t; t2 += warp.size()) {
            float att_t2 = att_bth[t2];
            float datt_t2 = datt_bth[t2];
            for(int k = 0; k < UNROLL; ++k) {
                if (t < t3 + k) continue;
                float indicator = t2 == (t3 + k) ? 1.0f : 0.0f;
                float local_derivative = att_t2 * (indicator - att_at_t3[k]);
                result[k] += local_derivative * datt_t2;
            }
        }

        for(int k = 0; k < UNROLL; ++k) {
            result[k] = cg::reduce(warp, result[k], cg::plus<float>());
        }
        if (warp.thread_rank() < UNROLL) {
            dpreatt_bth[t3 + warp.thread_rank()] = scale * result[warp.thread_rank()];
        }
    }
}

__global__ void softmax_autoregressive_backward_kernel5(float* __restrict__ dpreatt, const float* __restrict__ datt,
                                                        const float* __restrict__ att,
                                                        int B, int T, int C, int NH) {
    constexpr int UNROLL = 8;
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int t3 = UNROLL * (blockIdx.x * warp.meta_group_size() + warp.meta_group_rank());

    int idx = blockIdx.y * T * T;
    if (t3 >= T) { return; }

    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    for (int t = t3; t < T; t++) {
        float result[UNROLL] = {};
        const float* att_bth = att + idx + t * T;
        const float* datt_bth = datt + idx + t * T;
        float* dpreatt_bth = dpreatt + idx + t * T;

        float att_at_t3[UNROLL];
        for(int k = 0; k < UNROLL; ++k) {
            // if t < t3+k, we're out of bounds.
            // in that case, we don't care what we read, because later on,
            // we won't write the corresponding result. So just clip to
            // make sure this is a valid (in-bounds) memory access.
            att_at_t3[k] = att_bth[min(t, t3 + k)];
        }

        // the code below is actually just a for loop; except,
        // we have to do something special in one iteration in
        // the middle, and an if turned out to have significant
        // performance impact.
        // so we split the loop in three parts. Ugly, but effective.

        // the beginning/end loop does the same thing, so we write the code
        // just once in a lambda. In this step, we're guaranteed that
        // indicator == 0
        auto loop_step = [&](int t2){
            float p = att_bth[t2] * datt_bth[t2];
            for (int k = 0; k < UNROLL; ++k) {
                result[k] -= p * att_at_t3[k];
            }
        };

        // Now the actual loop.
        {
            // declare the loop iterator. Needs to be kept across the
            // three different parts, so it's not a local variable in
            // the for loop.
            int t2 = warp.thread_rank();

            // first part, as long as t2 < t3, indicator == 0
            for (; t2 < t3; t2 += warp.size()) {
                loop_step(t2);
            }

            // because k <= warp.size() (==32), the event that t3+k == t2
            // has to happen at this particular step.
            static_assert(UNROLL <= 32, "UNROLL is too large, this won't produce correct results.");
            if (t2 <= t) {
                float att_t2 = att_bth[t2];
                float datt_t2 = datt_bth[t2];
                float p = att_t2 * datt_t2;
                for (int k = 0; k < UNROLL; ++k) {
                    float indicator = t2 == (t3 + k) ? 1.0f : 0.0f;
                    result[k] += p * (indicator - att_at_t3[k]);
                }
                t2 += warp.size();
            }

            // rest of the loop, indicator == 0 again
            for (; t2 <= t; t2 += warp.size()) {
                loop_step(t2);
            }
        }

        for(int k = 0; k < UNROLL; ++k) {
            result[k] = cg::reduce(warp, result[k], cg::plus<float>());
        }

        // when storing, we need to check that this is actually a valid result.
        // here, warp.thread_rank() corresponds to `k` in the previous loops.
        if (warp.thread_rank() < UNROLL && t >= t3 + warp.thread_rank()) {
            dpreatt_bth[t3 + warp.thread_rank()] = scale * result[warp.thread_rank()];
        }
    }
}


// I want `BlockSize` to be statically known to the compiler, thus we get a template here.
// This kernel takes a step back, and looks at the original CPU code again. We have some simple outer loops
// That are independent, (b, t, h), and then the inner loops over (t2, t3) where we're combining elements -- this is
// where we can reuse data and be more efficient
// => handle b, t, h  through block indices; each block does all the work for the (t2, t3) loop cooperatively.
// Now we have two nested loops, and in the inner instruction, we combine indexing from both => this calls for
// loop tiling, and lifting some of the memory ops out of the loop.
// We're in luck here;  if we tile so that t3 is the outer loop, we can get a sinlge write op per result, AND also cache
// the t2-indexed part of the computation, which is the problematic one because it contains a multiplication that now we
// do not have to repeat over and over.
// => do an outer t3 loop where each thread gets one t3 index. Then, do an outer t2 loop in steps of BlockSize, and
// prepare BlockSize many elements for the inner loop. Here, each thread calculates one element and stores it in shmem.
// Then, in the inner t2 loop, each thread reads *all* the elements previously stored and does its computations.
// This way, we do 3*BlockSize loads, but BlockSize^2 computation steps => This kernel is now entirely compute bound.
// To fix up the compute issues, as above, we replace ifs in memory reading with min, and also split the inner loop
// into a large region where we don't have to calculate the indicator, and a small, costly region where we do.
template<int BlockSize>
__global__ void __launch_bounds__(BlockSize) softmax_autoregressive_backward_kernel6(float* dpreatt, const float* datt, const float* att,
                                                        int B, int T, int C, int NH) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    __shared__ float att_bth_s[BlockSize];

    int idx = blockIdx.y;
    int t = blockIdx.x;

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    const float* att_bth = att + t * T;
    const float* datt_bth = datt + t * T;
    float* dpreatt_bth = dpreatt + t * T;

    int block_steps = ceil_div(t+1, BlockSize);
    // very important: This loop condition needs to be the same for all threads.
    // even if a thread later on is not going to do any work, it needs to participate in the
    // data loading process!
    for (int t3f = 0; t3f < block_steps; ++t3f) {
        int t3 = t3f * BlockSize + block.thread_rank();
        float acc = 0.f;
        float at3 = att_bth[t3];
        for (int t2b = 0; t2b <= t; t2b += BlockSize) {
            int end = min(t + 1 - t2b, BlockSize);
            block.sync();
            {
                int t2i = block.thread_rank();
                int t2 = min(t, t2b + t2i);
                att_bth_s[t2i] = att_bth[t2] * datt_bth[t2];
            }

            block.sync();
            if(t3f * BlockSize == t2b) {
                for (int t2i = 0; t2i < end; t2i++) {
                    int t2 = t2b + t2i;
                    float indicator = t2 == t3 ? 1.0f : 0.0f;
                    acc += att_bth_s[t2i] * (indicator - at3);
                }
            } else {
                for (int t2i = 0; t2i < end; t2i++) {
                    acc +=  att_bth_s[t2i] * (0.f - at3);
                }
            }
        }
        dpreatt_bth[t3] = scale * acc;
    }
}

// Actually disentangling the loops and simplifying the resulting math gives us this pretty nice kernel.
template<int BlockSize>
__global__ void softmax_autoregressive_backward_kernel7(float* dpreatt, const float* datt, const float* att,
                                                        int B, int T, int C, float scale) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float block_acc[32];

    int idx = blockIdx.y;
    int t = blockIdx.x;

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    const float* att_bth = att + t * T;
    const float* datt_bth = datt + t * T;
    float* dpreatt_bth = dpreatt + t * T;

    if(warp.meta_group_rank() == 0) {
        block_acc[warp.thread_rank()] = 0;
    }

    float local_sum = 0;
    for(int t2 = block.thread_rank(); t2 <= t; t2 += BlockSize) {
        local_sum += att_bth[t2] * datt_bth[t2];
    }

    block_acc[warp.meta_group_rank()] = cg::reduce(warp, local_sum, cg::plus<float>{});
    block.sync();
    local_sum = cg::reduce(warp, block_acc[warp.thread_rank()], cg::plus<float>{});

    for (int t3 = block.thread_rank(); t3 <= t; t3 += BlockSize) {
        float acc = att_bth[t3] * (datt_bth[t3] - local_sum);
        dpreatt_bth[t3] = scale * acc;
    }
}

// The slightly less pretty version of kernel 7. Adding in all the dirty tricks that can give us a few more percent
//  - streaming memory access instructions
//  - reordering blocks to prevent tail effect
//  - multiple values of T per block
template<int BlockSize>
__global__ void softmax_autoregressive_backward_kernel8(float* dpreatt, const float* datt, const float* att,
                                                        int B, int T, int C, float scale) {
    namespace cg = cooperative_groups;
    constexpr int T_per_block = 4;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float block_acc[32];

    int idx = blockIdx.y;
    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block*blockIdx.x;

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    if (warp.meta_group_rank() == 0) {
        block_acc[warp.thread_rank()] = 0;
    }

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const float* att_bth = att + t * T;
        const float* datt_bth = datt + t * T;
        float* dpreatt_bth = dpreatt + t * T;

        float local_sum = 0;
        for (int t2 = block.thread_rank(); t2 <= t; t2 += BlockSize) {
            local_sum += att_bth[t2] * datt_bth[t2];
        }

        block_acc[warp.meta_group_rank()] = cg::reduce(warp, local_sum, cg::plus<float>{});
        block.sync();
        local_sum = cg::reduce(warp, block_acc[warp.thread_rank()], cg::plus<float>{});

        for (int t3 = block.thread_rank(); t3 <= t; t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            float acc = __ldcs(att_bth + t3) * (__ldcs(datt_bth + t3) - local_sum);
            __stcs(dpreatt_bth + t3, scale * acc);
        }
    }
}


// ----------------------------------------------------------------------------
// kernel launchers

// use att to store log l + m
void flash_attention_forward(float* out, float* inp, float* l,
                                int B, int T, int C, int NH) {
    // head size
    int HS = C / NH;

    // inp is (B, T, 3, NH, HS)
    // out is (B, T, NH, HS)
    // l is (B, T, NH)
//    int HS = C / NH; // head size
//    dim3 dimGrid1(NH, T, B);
//    dim3 dimBlock1(1);
//    flash_attention_forward_kernel0<<<dimGrid1, dimBlock1>>>(out, inp, l, B, T, NH, HS);



//      int T_r = 64;
//     dim3 dimGrid1(NH, T / 64, B);
//     dim3 dimBlock1(256);
//     int maxbytes1 = 65536;
//     cudaFuncSetAttribute(flash_attention_forward_kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes1);
//     flash_attention_forward_kernel1<<<dimGrid1, dimBlock1, maxbytes1>>>(out, inp, l, B, T, NH, HS);


//     dim3 dimGrid2(NH, T / 64, B);
//     dim3 dimBlock2(256);
//     int maxbytes2 = 49152;
//     cudaFuncSetAttribute(flash_attention_forward_kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes2);
//     flash_attention_forward_kernel2<<<dimGrid2, dimBlock2, maxbytes2>>>(out, inp, l, B, T, NH, HS);


    dim3 dimGrid3(NH, T / 128, B);
    dim3 dimBlock3(256);
    int maxbytes3 = 98304;
    cudaFuncSetAttribute(flash_attention_forward_kernel3, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes3);
    flash_attention_forward_kernel3<<<dimGrid3, dimBlock3, maxbytes3>>>(out, inp, l, B, T, NH, HS);

    cudaCheck(cudaGetLastError());

}


void flash_attention_backward(float *dinp, float* inp, float* dout, float* out, float* l, float* d,
                                int B, int T, int C, int NH) {

    int HS = C / NH; // head size


    // flash attention backward v0
//     dim3 dimGrid0(NH, T, B);
//     dim3 dimBlock0(1);
//     flash_attention_backward_kernel0<<<dimGrid0, dimBlock0>>>(dinp, inp, dout, out, l, B, T, NH, HS);


    // flash attention backward v1

    // preprocess D = rowsum(dO * O)
    dim3 dimGrid_preprocessing1(NH, T, B);
    dim3 dimBlock_preprocessing1(1);
    flash_attention_backward_preprocessing_kernel1<<<dimGrid_preprocessing1, dimBlock_preprocessing1>>>(d, dout, out, B, T, NH, HS);

//     dim3 dimGrid1(NH, T / 32, B);
//     dim3 dimBlock1(128);
//     int maxbytes1 = 98304;
//     cudaFuncSetAttribute(flash_attention_backward_kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes1);
//     flash_attention_backward_kernel1<<<dimGrid1, dimBlock1, maxbytes1>>>(dinp, inp, dout, out, l, d, B, T, NH, HS);


    dim3 dimGrid2(NH, T / 64, B);
    dim3 dimBlock2(256);
    int maxbytes2 = 65536;
    cudaFuncSetAttribute(flash_attention_backward_kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes2);
    flash_attention_backward_kernel2<<<dimGrid2, dimBlock2, maxbytes2>>>(dinp, inp, dout, out, l, d, B, T, NH, HS);


    dim3 dimGrid3(NH, T / 128, B);
    dim3 dimBlock3(256);
    int maxbytes3 = 65536;
    cudaFuncSetAttribute(flash_attention_backward_kernel3, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes3);
    flash_attention_backward_kernel3<<<dimGrid3, dimBlock3, maxbytes3>>>(dinp, inp, dout, out, l, d, B, T, NH, HS);


    cudaCheck(cudaGetLastError());
}

// attention forward pass kernel
void attention_forward(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     T, T, HS,
                                     &alpha,
                                     k, HS, T * HS,
                                     q, HS, T * HS,
                                     &beta,
                                     preatt, T, T * T,
                                     B * NH));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int softmax_block_size = 256;
    int grid_size = ceil_div(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);

    // new approach: first cuBLAS another batched matmul
    // vaccum = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     HS, T, T,
                                     &alpha,
                                     v, HS, T * HS,
                                     att, T, T * T,
                                     &beta,
                                     vaccum, HS, T * HS,
                                     B * NH));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
}

void launch_softmax_1(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(T, block_size);
    softmax_autoregressive_backward_kernel1<<<dim3(num_blocks, B*NH), block_size>>>(dpreatt, datt, att, B, T, C, NH);
}

void launch_softmax_2(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(T, block_size);
    softmax_autoregressive_backward_kernel2<<<dim3(num_blocks, B*NH), block_size>>>(dpreatt, datt, att, B, T, C, NH);
}

void launch_softmax_3(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(32*T, block_size);
    softmax_autoregressive_backward_kernel3<<<dim3(num_blocks, B*NH), block_size>>>(dpreatt, datt, att, B, T, C, NH);
}

void launch_softmax_4(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(32/8*T, block_size);
    softmax_autoregressive_backward_kernel4<<<dim3(num_blocks, B*NH), block_size>>>(dpreatt, datt, att, B, T, C, NH);
}

void launch_softmax_5(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(32/8*T, block_size);
    softmax_autoregressive_backward_kernel5<<<dim3(num_blocks, B*NH), block_size>>>(dpreatt, datt, att, B, T, C, NH);
}

template<class Launcher>
void dispatch_launch(Launcher&& launch, int block_size) {
    switch(block_size) {
        case 32:
            return launch(std::integral_constant<int, 32>{});
        case 64:
            return launch(std::integral_constant<int, 64>{});
        case 128:
            return launch(std::integral_constant<int, 128>{});
        case 256:
            return launch(std::integral_constant<int, 256>{});
        case 512:
            return launch(std::integral_constant<int, 512>{});
        case 1024:
            return launch(std::integral_constant<int, 1024>{});
        default:
            assert(false && "Invalid block size");
    }
}

void launch_softmax_6(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    auto launch = [&](auto int_const) {
        softmax_autoregressive_backward_kernel6<int_const.value><<<dim3(T, B * NH), int_const.value>>>(dpreatt, datt, att, B, T, C, NH);
    };
    dispatch_launch(launch, block_size);
}

void launch_softmax_7(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    auto launch = [&](auto int_const) {
        constexpr int block_size = int_const.value;
        softmax_autoregressive_backward_kernel7<block_size><<<dim3(T, B * NH), block_size>>>
                                                              (dpreatt, datt, att, B, T, C, scale);
    };
    dispatch_launch(launch, block_size);
}

void launch_softmax_8(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    auto launch = [&](auto int_const) {
        constexpr int block_size = int_const.value;
        softmax_autoregressive_backward_kernel8<block_size><<<dim3(T / 4, B * NH), block_size>>>
                                                              (dpreatt, datt, att, B, T, C, scale);
    };
    dispatch_launch(launch, block_size);
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
template<class SoftmaxKernel>
void attention_backward1(float* dinp, float* dqkvr, float* dpreatt, float* datt, float* dvaccum,
                        const float* dout,
                        const float* inp, const float* qkvr, const float* preatt, const float* att, const float* vaccum,
                        int B, int T, int C, int NH,
                        SoftmaxKernel softmax_autoregressive_backward,
                        const int block_size) {
    int HS = C / NH; // head size
    const float alpha = 1.0f;
    const float beta = 1.0f; // note beta = 1.0f so that we accumulate gradients (+=)
    // unpack convenience pointers into q, k, v
    const float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    float *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;

    // backward through the unpermute operation
    int num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size>>>(dvaccum, dout, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    // backward into datt
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            T, T, HS,
                            &alpha,
                            v, HS, T * HS,
                            dvaccum, HS, T * HS,
                            &beta,
                            datt, T, T * T,
                            B * NH));

    // backward into dv
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            HS, T, T,
            &alpha,
            dvaccum, HS, T * HS,
            att, T, T * T,
            &beta,
            dv, HS, T * HS,
            B * NH));

    // backward into preatt
    softmax_autoregressive_backward(dpreatt, datt, att, B, T, C, NH, block_size);
    cudaCheck(cudaGetLastError());

    // backward into q
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            HS, T, T,
                            &alpha,
                            k, HS, T * HS,
                            dpreatt, T, T * T,
                            &beta,
                            dq, HS, T * HS,
                            B * NH));
    // backward into k
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            HS, T, T,
                            &alpha,
                            q, HS, T * HS,
                            dpreatt, T, T * T,
                            &beta,
                            dk, HS, T * HS,
                            B * NH));

    // backward into inp
    num_blocks = ceil_div(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void attention_backward(int kernel_num,
                        float* dinp, float* dqkvr, float* dpreatt, float* datt, float* dvaccum,
                        const float* dout,
                        const float* inp, const float* qkvr, const float* preatt, const float* att, const float* vaccum,
                        int B, int T, int C, int NH,
                        const int block_size) {
    switch (kernel_num) {
        case 1:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_1, block_size);
            break;
        case 2:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_2, block_size);
            break;
        case 3:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_3, block_size);
            break;
        case 4:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_4, block_size);
            break;
        case 5:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_5, block_size);
            break;
        case 6:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_6, block_size);
            break;
        case 7:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_7, block_size);
            break;
        case 8:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_8, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    setup_main();

    // hyperparameters
//     int B = 4;
//     int T = 1024;
//     int C = 768;
//     int NH = 12;
   int B = 1;
   int T = 128;
   int C = 64;
   int NH = 1;

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // create the host memory for the forward pass
    float* inp = make_random_float(B * T * 3 * C);
    float* qkvr = (float*)malloc(B * T * 3 * C * sizeof(float));
    float* preatt = (float*)malloc(B * NH * T * T * sizeof(float));
    float* att = (float*)malloc(B * NH * T * T * sizeof(float));
    float* vaccum = (float*)malloc(B * T * C * sizeof(float));
    float* out = (float*)malloc(B * T * C * sizeof(float));

    // execute the forward pass on the CPU
    attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);

    // create device memory for the forward pass
    float *d_inp, *d_qkvr, *d_preatt, *d_att, *d_vaccum, *d_out, *d_l;
    cudaCheck(cudaMalloc(&d_inp, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_qkvr, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_preatt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_vaccum, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_l, B * T * NH * sizeof(float)));
    // copy over the input
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * 3 * C * sizeof(float), cudaMemcpyHostToDevice));

    // execute the forward pass on the GPU
    const int block_size = 256;
    attention_forward(d_out, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T, C, NH, block_size);
    flash_attention_forward(d_out, d_inp, d_l, B, T, C, NH);

    // check that preatt, att, and out match between the CPU and GPU versions
    printf("Checking the forward pass CPU <-> GPU...\n");
    //printf("[preatt]\n"); validate_result(d_preatt, preatt, "preatt", B * T * C, 5e-3f);
    //printf("[att]\n");    validate_result(d_att, att, "att", B * T * C, 1e-3f);
    printf("[out]\n");    validate_result(d_out, out, "out", B * T * C, 1e-3f);

    // set up the memory for the backward pass
    float* dout = make_random_float(B * T * C); // the gradients on the output
    float* dinp = make_zeros_float(B * T * 3 * C); // zeros for all else, to += into
    float* dpreatt = make_zeros_float(B * NH * T * T);
    float* datt = make_zeros_float(B * NH * T * T);

    // call backward() on the CPU to get our reference gradients
    attention_backward_cpu(dinp, dpreatt, datt, dout, inp, att, B, T, C, NH);




    // create device memory for the backward pass
    float *d_dinp, *d_dqkvr, *d_dpreatt, *d_datt, *d_dvaccum, *d_dout, *d_d;
    cudaCheck(cudaMalloc(&d_dinp, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dqkvr, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dpreatt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_datt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dvaccum, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_d, B * T * NH * sizeof(float)));
    // copy over the dout gradients that starts the backprop chain
    cudaCheck(cudaMemcpy(d_dout, dout, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    // memset all the other memory to zeros, to += into
    cudaCheck(cudaMemset(d_dinp, 0, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMemset(d_dqkvr, 0, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMemset(d_dpreatt, 0, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMemset(d_datt, 0, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMemset(d_dvaccum, 0, B * T * C * sizeof(float)));



    // call backward() on the GPU
    attention_backward(kernel_num, d_dinp, d_dqkvr, d_dpreatt, d_datt, d_dvaccum,
                       d_dout, d_inp, d_qkvr, d_preatt, d_att, d_vaccum,
                       B, T, C, NH, block_size);
    cudaCheck(cudaMemset(d_dinp, 0, B * T * 3 * C * sizeof(float)));
    flash_attention_backward(d_dinp, d_inp, d_dout, d_out, d_l, d_d, B, T, C, NH);

//    for (int i=0; i <  B * T * 3 * C; i++) {
//        printf("i = %d, cpu = %f\n", i, dinp[i], d_dinp[i]);
//    }

    // check that the gradients match between the CPU and GPU versions
    // note that we will only check the correctness at [att, preatt, inp]
    // the gradients at qkvr and vaccum will remain unchecked, but are
    // assumed to be correct if the other gradients are correct
    printf("Checking the backward pass CPU <-> GPU...\n");
    //printf("[datt]\n");    validate_result(d_datt, datt, "datt", B * NH * T * T, 5e-3f);
    //printf("[dpreatt]\n"); validate_result(d_dpreatt, dpreatt, "dpreatt", B * NH * T * T, 1e-3f);
    printf("[dinp]\n");    validate_result(d_dinp, dinp, "dinp", B * T * 3 * C, 1e-3f);



    // also let's manually step through the gradients here
    float* h_dinp = (float*)malloc(B * T * 3 * C * sizeof(float));
    cudaCheck(cudaMemcpy(h_dinp, d_dinp, B * T * 3 * C * sizeof(float), cudaMemcpyDeviceToHost));
    int num_match = 0;
    int num_no_match = 0;
    int num_zero_grad = 0;
    int HS = C / NH;
    for (int i = 0; i < B * T * 3 * C; i++) {

        // the dimensions of inp are (B, T, 3, NH, HS)
        // where B = batch, T = time, 3 = qkv, NH = num heads, HS = head size
        // unpack the individual b,t,qkvix,h,c indices
        int ix = i;
        int c = ix % HS;
        ix /= HS;
        int h = ix % NH;
        ix /= NH;
        int qkvix = ix % 3;
        ix /= 3;
        int t = ix % T;
        ix /= T;
        int b = ix;

        float diff = fabs(dinp[i] - h_dinp[i]);

        // attempt to index at random
        if (b == 1 && t == 5 && c == 23 && h == 2) {
            printf("ix %5d [b=%4d, t=%4d, qkv=%4d, nh=%4d, hs=%4d]: ref: %f gpu: %f\n", i, b, t, qkvix, h, c, dinp[i], h_dinp[i]);
        }

        if (diff > 1e-4f) {
            num_no_match++;
        } else {
            num_match++;
        }

        if (dinp[i] == 0.0f) {
            num_zero_grad++;
        }
    }
    printf("Number of matching gradients: %d (%.2f%% of total)\n", num_match, 100*(float)num_match / (B * T * 3 * C));
    printf("Number of non-matching gradients: %d (%.2f%% of total)\n", num_no_match, 100*(float)num_no_match / (B * T * 3 * C));
    printf("Number of gradients that are exactly zero: %d (%.2f%% of total)\n", num_zero_grad, 100*(float)num_zero_grad / (B * T * 3 * C));

    // final verdict
//    printf("All results match. Starting benchmarks.\n\n");
//
//    // benchmark speed of the kernel
//    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
//    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
//        int block_size = block_sizes[j];
//        int repeat_times = 10;
//        float elapsed_time = benchmark_kernel(repeat_times, attention_backward,
//                                              kernel_num, d_dinp, d_dqkvr, d_dpreatt, d_datt, d_dvaccum,
//                                              d_dout, d_inp, d_qkvr, d_preatt, d_att, d_vaccum,
//                                              B, T, C, NH, block_size);
//
//        printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
//    }

    // free memory
    free(inp);
    free(qkvr);
    free(preatt);
    free(att);
    free(vaccum);
    free(out);
    free(dout);
    free(dinp);
    free(dpreatt);
    free(datt);
    free(h_dinp);
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_preatt));
    cudaCheck(cudaFree(d_att));
    cudaCheck(cudaFree(d_vaccum));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_dqkvr));
    cudaCheck(cudaFree(d_dpreatt));
    cudaCheck(cudaFree(d_datt));
    cudaCheck(cudaFree(d_dvaccum));
    cudaCheck(cudaFree(d_dout));
    cublasDestroy(cublas_handle);
    return 0;
}