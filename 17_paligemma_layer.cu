/*
 * CUDA 编程教程 - 第15课：PaliGemma 模型核心组件 (RoPE & GeGLU)
 *
 * PaliGemma 是 Google 推出的视觉-语言模型 (VLM)，结合了 SigLIP (视觉) 和 Gemma (语言)。
 * 其语言部分 (Gemma) 具有一些独特的架构特征，与标准 Transformer 不同。
 *
 * 本教程将实现 PaliGemma/Gemma 的两个关键 CUDA 组件：
 * 1. RoPE (Rotary Positional Embeddings): 旋转位置编码，用于 Attention。
 * 2. GeGLU (Gated GeLU): 门控激活函数，用于 MLP 层。
 *
 * Gemma Decoder 层结构:
 * Input -> RMSNorm -> Attention (w/ RoPE) -> Residual
 * Input -> RMSNorm -> MLP (GeGLU) -> Residual
 *
 * 编译和运行:
 * nvcc 15_paligemma_layer.cu -o 15_paligemma_layer -O3
 * ./15_paligemma_layer
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// --- 超参数配置 (模拟 Gemma 2B 规模) ---
#define SEQ_LEN 16
#define DIM 2048           // 模型维度
#define HEAD_DIM 256       // 注意力头维度 (Gemma 使用较大的 Head Dim)
#define N_HEADS 8          // 头数 (DIM = N_HEADS * HEAD_DIM)
#define HIDDEN_DIM 16384   // MLP 中间层维度 (通常很大)

#define THREADS_PER_BLOCK 256

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while(0)

// ==========================================
// 1. 核心内核 (Kernels)
// ==========================================

// 1.1 RMSNorm (Gemma 风格: output = x * w * inv_rms)
// 注意：Gemma 的 RMSNorm 权重通常初始化为 0 并在计算时 +1 (unit offset)，
// 但这里为了通用性，我们假设权重已经处理好 (即标准 RMSNorm)。
__global__ void rms_norm_kernel(const float* __restrict__ input, 
                                const float* __restrict__ weight, 
                                float* __restrict__ output, 
                                int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * n + tid;

    __shared__ float s_sum_sq;
    if (tid == 0) s_sum_sq = 0.0f;
    __syncthreads();

    float val = (tid < n) ? input[idx] : 0.0f;
    atomicAdd(&s_sum_sq, val * val);
    __syncthreads();

    float inv_rms = rsqrtf(s_sum_sq / n + 1e-6f); // Gemma 使用 1e-6 epsilon

    if (tid < n) {
        output[idx] = val * inv_rms * weight[tid];
    }
}

// 1.2 RoPE (Rotary Positional Embeddings)
// PaliGemma/Gemma 使用 RoPE 来注入位置信息。
// 公式: 
// x1_new = x1 * cos(theta) - x2 * sin(theta)
// x2_new = x1 * sin(theta) + x2 * cos(theta)
// 这里的 theta 取决于位置 (pos) 和维度索引 (i)。
// 为了简化，我们假设预先计算好了 cos_theta 和 sin_theta 表。
// q_k_input: [Seq, Heads, HeadDim]
__global__ void rope_kernel(float* __restrict__ data, 
                            const float* __restrict__ cos_cache, 
                            const float* __restrict__ sin_cache, 
                            int head_dim, int seq_len, int n_heads) {
    // 这里的并行策略：
    // blockIdx.x: 序列位置 (0 ~ seq_len-1)
    // blockIdx.y: 注意力头 (0 ~ n_heads-1)
    // threadIdx.x: 维度对索引 (0 ~ head_dim/2 - 1)
    
    int pos = blockIdx.x;
    int head = blockIdx.y;
    int pair_idx = threadIdx.x; // 处理第 pair_idx 对元素 (2 * pair_idx, 2 * pair_idx + 1)

    if (pair_idx >= head_dim / 2) return;

    // 计算数据在全局内存中的偏移
    // Data layout: [Seq, Heads, HeadDim]
    int base_idx = pos * (n_heads * head_dim) + head * head_dim;
    
    int idx1 = base_idx + 2 * pair_idx;
    int idx2 = base_idx + 2 * pair_idx + 1;

    float x1 = data[idx1];
    float x2 = data[idx2];

    // 读取预计算的 cos/sin 值
    // Cache layout: [Seq, HeadDim/2]
    int freq_idx = pos * (head_dim / 2) + pair_idx;
    float c = cos_cache[freq_idx];
    float s = sin_cache[freq_idx];

    // 应用旋转
    float out1 = x1 * c - x2 * s;
    float out2 = x1 * s + x2 * c;

    data[idx1] = out1;
    data[idx2] = out2;
}

// 1.3 GeGLU (Gated GeLU)
// Gemma 的 MLP 使用 GeGLU 激活。
// 输入通常是两个线性层的输出：Gate 和 Value。
// Output = GeLU(Gate) * Value
// 这里的 input_gate 和 input_val 维度都是 [Seq, HiddenDim]
__global__ void geglu_kernel(const float* __restrict__ input_gate, 
                             const float* __restrict__ input_val, 
                             float* __restrict__ output, 
                             int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float g = input_gate[idx];
    float v = input_val[idx];

    // 标准 Tanh 近似 GeLU
    // GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float c1 = 0.7978845608f; 
    const float c2 = 0.044715f;
    float gelu_g = 0.5f * g * (1.0f + tanh(c1 * (g + c2 * g * g * g)));

    // GeGLU = GeLU(Gate) * Value
    output[idx] = gelu_g * v;
}

// 1.4 简单的矩阵乘法 (用于演示)
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ==========================================
// 2. PaliGemma 层模拟
// ==========================================

void run_paligemma_layer_demo() {
    // 1. 准备数据
    size_t input_size = SEQ_LEN * DIM * sizeof(float);
    size_t qk_size = SEQ_LEN * N_HEADS * HEAD_DIM * sizeof(float);
    size_t mlp_size = SEQ_LEN * HIDDEN_DIM * sizeof(float);
    size_t rope_cache_size = SEQ_LEN * (HEAD_DIM / 2) * sizeof(float);

    float *d_input, *d_norm_out;
    float *d_q, *d_k; // Query, Key (需要 RoPE)
    float *d_gate, *d_val, *d_mlp_out; // MLP
    float *d_cos, *d_sin; // RoPE Cache

    CHECK_CUDA(cudaMalloc(&d_input, input_size));
    CHECK_CUDA(cudaMalloc(&d_norm_out, input_size));
    CHECK_CUDA(cudaMalloc(&d_q, qk_size));
    CHECK_CUDA(cudaMalloc(&d_k, qk_size));
    CHECK_CUDA(cudaMalloc(&d_gate, mlp_size));
    CHECK_CUDA(cudaMalloc(&d_val, mlp_size));
    CHECK_CUDA(cudaMalloc(&d_mlp_out, mlp_size));
    CHECK_CUDA(cudaMalloc(&d_cos, rope_cache_size));
    CHECK_CUDA(cudaMalloc(&d_sin, rope_cache_size));

    // 初始化 RoPE Cache (模拟)
    // 实际中根据 theta = 10000^(-2i/d) 计算
    float *h_cos = (float*)malloc(rope_cache_size);
    float *h_sin = (float*)malloc(rope_cache_size);
    for(int p=0; p<SEQ_LEN; p++) {
        for(int i=0; i<HEAD_DIM/2; i++) {
            float theta = 1.0f / powf(10000.0f, 2.0f*i/HEAD_DIM);
            float angle = p * theta;
            h_cos[p*(HEAD_DIM/2)+i] = cosf(angle);
            h_sin[p*(HEAD_DIM/2)+i] = sinf(angle);
        }
    }
    CHECK_CUDA(cudaMemcpy(d_cos, h_cos, rope_cache_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sin, h_sin, rope_cache_size, cudaMemcpyHostToDevice));

    // 初始化输入数据 (随机)
    // ... (省略初始化 d_input, d_q, d_k, d_gate, d_val 的代码) ...

    printf("Running PaliGemma Layer Components...\n");

    // --- 步骤 1: RoPE (应用于 Q 和 K) ---
    // Grid: [Seq, Heads], Block: [HeadDim/2]
    dim3 rope_grid(SEQ_LEN, N_HEADS);
    dim3 rope_block(HEAD_DIM / 2);
    
    printf("1. Applying RoPE to Query and Key...\n");
    rope_kernel<<<rope_grid, rope_block>>>(d_q, d_cos, d_sin, HEAD_DIM, SEQ_LEN, N_HEADS);
    rope_kernel<<<rope_grid, rope_block>>>(d_k, d_cos, d_sin, HEAD_DIM, SEQ_LEN, N_HEADS);
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- 步骤 2: GeGLU (MLP 激活) ---
    // 假设已经完成了 Gate_Proj 和 Up_Proj 的矩阵乘法，结果在 d_gate 和 d_val 中
    int total_elements = SEQ_LEN * HIDDEN_DIM;
    int num_blocks = (total_elements + 255) / 256;
    
    printf("2. Applying GeGLU Activation...\n");
    geglu_kernel<<<num_blocks, 256>>>(d_gate, d_val, d_mlp_out, total_elements);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("PaliGemma components executed successfully.\n");

    // 清理
    free(h_cos); free(h_sin);
    cudaFree(d_input); cudaFree(d_norm_out);
    cudaFree(d_q); cudaFree(d_k);
    cudaFree(d_gate); cudaFree(d_val); cudaFree(d_mlp_out);
    cudaFree(d_cos); cudaFree(d_sin);
}

int main() {
    printf("=== PaliGemma / Gemma CUDA Implementation Tutorial ===\n");
    run_paligemma_layer_demo();
    return 0;
}
