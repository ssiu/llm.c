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
        x *= 1.0 / sqrtf(HS);
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

        //update constants
        m_old = m;
        d_old = d;

        //update k,v offset
        gK += 3 * NH * HS;
        gV += 3 * NH * HS;
    }

    gL[0] = logf(d) + m;

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
#define sQ(i,j) sQ[(i) * HS + (j)]
#define sK(i,j) sK[(i) * HS + (j)]
#define sV(i,j) sV[(i) * HS + (j)]
#define sP(i,j) sP[(i) * HS + (j)]
#define FLOAT4(addr) reinterpret_cast<float4*>(&(addr))[0]
__global__ void flash_attention_forward_kernel1(float* out, float* inp, float* l,
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
    int l_global_offset = blockIdx.z * T * NH + blockIdx.y * NH + blockIdx.x;

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

    if (threadIdx.x==0) {
        for (int i=0;i<64;i++) {
            for (int j=0;j<64;j++) {
                printf("%f ", sQ(i,j));
            }
            printf("\n");
        }
    }



    __syncthreads();

    // main loop
    float rQ[4];
    float rK[4];
    float rV[4];
    float tS[4][4] = {0.0f};
    float (&tP)[4][4] = tS;
    float rP[4];
    float rO[4][4] = {0.0f};
    float rM_old[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    float rM[4];
    float rL_old[4] = {0.0f};
    float rL[4] = {0.0f};
    // now we need to put the auto regressive mask, what should it be
    // only need to check when kv_tile = blockIdx.y
    for (int kv_tile = 0; kv_tile <= blockIdx.y; kv_tile++) {
        if (threadIdx.x==0) {
            printf("HIHIHIHELO");
        }
        // load gK to sK, need to first transpose
        float rK_shared[4];
        for (int i = 0; i < 4; i++) {
            FLOAT4(rK_shared[0]) = FLOAT4(gK(warp_row + thread_row , thread_col));
            for (int j=0; j < 4; j++) {
                sK(thread_col + j, warp_row + thread_row + i) = rK_shared[j];
            }
        }
        // load gV to sV
        for (int i = 0; i < 4; i++) {
	        FLOAT4(sV(warp_row + thread_row + i, thread_col)) = FLOAT4(gV(warp_row + thread_row + i, thread_col));
        }

        __syncthreads();
        //
        // compute rS
        //
        for (int k_fragment = 0; k_fragment < HS; k_fragment++) {

            for (int i = 0; i < 4; i++) {
                rQ[i] = sQ(warp_row + thread_row + i, k_fragment);
                rK[i] = sK(k_fragment, thread_col + i);
            }

            for (int i = 0; i < 4; i++) {
                for (int j = 0; i < 4; j++) {
                    if (blockIdx.y == kv_tile && warp_row + thread_row + i < thread_col + j) {
                        tS[i][j] = rQ[i] * rK[j];
                    } else {
                        // apply casual mask
                        tS[i][j] = -FLT_MAX;
                    }

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

        // compute l

        for (int i=0;i<4;i++) {
            rL[i] = exp(rM_old[i] - rM[i]) * rL_old[i];
            for (int j=0;j<4;j++){
                rL[i] += tP[i][j];
            }
        }

        //store to sP
        for (int i=0;i<4;i++) {
            for (int j=0;j<4;j++) {
                sP(warp_row + thread_row + i , thread_col + j ) = tP[i][j];
            }
        }


        //
        // compute O
        //

        // first rescale O by exp(m_old - m)
        for (int i=0; i<4; i++) {
            for (int j=0;j<4;j++) {
                rO[i][j] = expf(rM[i] - rM_old[i]) * rO[i][j];
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

    // store l back to gL
    if (lane_id == 0 or lane_id == 16) {
        for (int i = 0; i < 4; i++) {
            gL(warp_row + thread_row + i) = rM[i] + logf(rL[i]);
        }
    }
    // store rO to gO
    for (int i=0; i < 4; i++) {
        for (int j=0;j<4;j++) {
            gO(warp_row + thread_row + i, thread_col + j) = rO[i][j];
        }
    }


}

// inp/dinp are (B, T, 3C), (B, T, 3, NH, HS) Q,K,V
//
// att/datt/dpreatt are (B, NH, T, T)
// dout is (B, T, C)
// l is (B, T, NH)
// blockDim.x = NH
// blockDim.y = T
// blockDim.z = B

__global__ void flash_attention_backward_kernel0(float* dinp, float* inp, float* dout, float* out, float* l,
                                int B, int T, int NH, int HS) {
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


    // inp is (B, T, 3, NH, HS)
    // out is (B, T, NH, HS)
    // l is (B, T, NH)
//    int HS = C / NH; // head size
//    dim3 dimGrid(NH, T, B);
//    dim3 dimBlock(1);
//    flash_attention_forward_kernel0<<<dimGrid, dimBlock>>>(out, inp, l, B, T, NH, HS);
    //int T_r = 64;
    int HS = C / NH; // head size
    dim3 dimGrid(NH, T / 64, B);
    dim3 dimBlock(256);
    int maxbytes = 65536;
    cudaFuncSetAttribute(flash_attention_forward_kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    flash_attention_forward_kernel1<<<dimGrid, dimBlock, maxbytes>>>(out, inp, l, B, T, NH, HS);

    cudaCheck(cudaGetLastError());

}


void flash_attention_backward(float *dinp, float* inp, float* dout, float* out, float* l,
                                int B, int T, int C, int NH) {

    int HS = C / NH; // head size
    dim3 dimGrid(NH, T, B);
    dim3 dimBlock(1);
    flash_attention_backward_kernel0<<<dimGrid, dimBlock>>>(dinp, inp, dout, out, l, B, T, NH, HS);

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
    int B = 1;
    int T = 64;
//    int C = 768;
//    int NH = 12;
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
   // attention_forward(d_out, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T, C, NH, block_size);
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
    float *d_dinp, *d_dqkvr, *d_dpreatt, *d_datt, *d_dvaccum, *d_dout;
    cudaCheck(cudaMalloc(&d_dinp, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dqkvr, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dpreatt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_datt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dvaccum, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(float)));
    // copy over the dout gradients that starts the backprop chain
    cudaCheck(cudaMemcpy(d_dout, dout, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    // memset all the other memory to zeros, to += into
    cudaCheck(cudaMemset(d_dinp, 0, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMemset(d_dqkvr, 0, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMemset(d_dpreatt, 0, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMemset(d_datt, 0, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMemset(d_dvaccum, 0, B * T * C * sizeof(float)));


    // call backward() on the GPU
//    attention_backward(kernel_num, d_dinp, d_dqkvr, d_dpreatt, d_datt, d_dvaccum,
//                       d_dout, d_inp, d_qkvr, d_preatt, d_att, d_vaccum,
//                       B, T, C, NH, block_size);
    flash_attention_backward(d_dinp, d_inp, d_dout, d_out, d_l, B, T, C, NH);

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
    printf("All results match. Starting benchmarks.\n\n");

    // benchmark speed of the kernel
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 10;
        float elapsed_time = benchmark_kernel(repeat_times, attention_backward,
                                              kernel_num, d_dinp, d_dqkvr, d_dpreatt, d_datt, d_dvaccum,
                                              d_dout, d_inp, d_qkvr, d_preatt, d_att, d_vaccum,
                                              B, T, C, NH, block_size);

        printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
    }

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