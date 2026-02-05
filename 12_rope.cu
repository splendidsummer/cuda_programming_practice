/*
 * CUDA 编程教程 - 第12课：RoPE (Rotary Positional Embedding) 旋转位置编码
 *
 * RoPE 是现代 LLM (如 LLaMA, PaLM) 中广泛使用的位置编码方式。
 * 它通过旋转复数空间中的向量来注入绝对和相对位置信息。
 *
 * 核心操作：
 * 对于向量中的每对元素 (x1, x2)，根据其位置 m 和频率 theta 进行旋转：
 * | x1' | = | cos(m*theta)  -sin(m*theta) | | x1 |
 * | x2' |   | sin(m*theta)   cos(m*theta) | | x2 |
 *
 * 编译和运行:
 * nvcc 12_rope.cu -o 12_rope -O3
 * ./12_rope
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024      // 序列长度 (Sequence Length)
#define D 128       // 维度 (Head Dimension)，必须是偶数
#define HEADS 8     // 注意力头数
#define THREADS_PER_BLOCK 256

// RoPE 内核
// 每个线程处理一个 Head 的一部分
__global__ void rope_kernel(float *Q, float *K, int seq_len, int head_dim, int num_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * num_heads * head_dim;

    if (idx >= total_elements / 2) return; // 每个线程处理一对元素 (2个)，所以总线程数减半

    // 计算当前处理的逻辑坐标
    // 这里的 idx 是 "对" 的索引
    int half_dim = head_dim / 2;
    
    // 1. 确定当前处理的是哪个 token (seq_idx) 和哪个 head (head_idx)
    // 布局假设: [batch(1), seq_len, num_heads, head_dim]
    // 扁平化索引: seq_idx * (num_heads * head_dim) + head_idx * head_dim + dim_idx
    
    int element_idx_in_seq = idx % (num_heads * half_dim); // 在当前 token 内的偏移 (pair index)
    int seq_idx = idx / (num_heads * half_dim);            // 当前 token 的位置 m
    
    int head_idx = element_idx_in_seq / half_dim;          // 当前 head
    int dim_pair_idx = element_idx_in_seq % half_dim;      // 在 head 内的 pair 索引 (0 到 D/2 - 1)

    // 2. 计算频率 theta
    // formula: theta_i = 10000 ^ (-2 * (i-1) / d)
    // 在 RoPE 中，通常将维度分为前半部分和后半部分，或者交错配对。
    // 这里我们使用标准的实现：相邻配对或半半配对。
    // LLaMA 通常使用极坐标形式，这里演示最直观的实现：
    // theta = 1.0 / pow(10000.0, 2.0 * dim_pair_idx / head_dim)
    
    float freq = 1.0f / powf(10000.0f, 2.0f * dim_pair_idx / (float)head_dim);
    float angle = (float)seq_idx * freq;
    
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // 3. 读取 Q 和 K 的值
    // 我们需要读取 x[2*i] 和 x[2*i+1] 或者 x[i] 和 x[i + D/2]
    // LLaMA 的实现通常是 x[i] 和 x[i + D/2] (Rotary on half dimension)
    // 但为了简单直观，这里演示相邻元素旋转: x[2*k] 和 x[2*k+1]
    
    int base_offset = seq_idx * num_heads * head_dim + head_idx * head_dim;
    int idx1 = base_offset + 2 * dim_pair_idx;
    int idx2 = base_offset + 2 * dim_pair_idx + 1;

    float q1 = Q[idx1];
    float q2 = Q[idx2];
    float k1 = K[idx1];
    float k2 = K[idx2];

    // 4. 应用旋转
    // | q1' | = | cos -sin | | q1 |
    // | q2' |   | sin  cos | | q2 |
    
    Q[idx1] = q1 * cos_val - q2 * sin_val;
    Q[idx2] = q1 * sin_val + q2 * cos_val;
    
    K[idx1] = k1 * cos_val - k2 * sin_val;
    K[idx2] = k1 * sin_val + k2 * cos_val;
}

// ==========================================
// 优化版 Kernel: 预计算表 + float2 向量化
// ==========================================
__global__ void rope_kernel_opt(float2 *Q, float2 *K, const float2 *CosSinTable, int seq_len, int head_dim, int num_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_pairs = seq_len * num_heads * half_dim;

    if (idx >= total_pairs) return;

    // 1. 计算逻辑坐标
    // idx 对应的是 float2 的索引
    int element_idx_in_seq = idx % (num_heads * half_dim);
    int seq_idx = idx / (num_heads * half_dim);
    int dim_pair_idx = element_idx_in_seq % half_dim;

    // 2. 查表读取 Cos/Sin
    // 表的大小是 [seq_len, head_dim/2]
    // 所有的 Head 共享同一个表，所以不需要 head_idx
    int table_idx = seq_idx * half_dim + dim_pair_idx;
    
    // 使用 __ldg 强制走只读缓存 (Read-Only Cache)，大幅提升频繁读取常量的速度
    float2 cs = __ldg(&CosSinTable[table_idx]);
    float cos_val = cs.x;
    float sin_val = cs.y;

    // 3. 向量化读取 Q 和 K (一次读 64位)
    float2 q_pair = Q[idx];
    float2 k_pair = K[idx];

    // 4. 旋转计算 (纯乘加运算，极快)
    float2 q_out, k_out;
    
    q_out.x = q_pair.x * cos_val - q_pair.y * sin_val;
    q_out.y = q_pair.x * sin_val + q_pair.y * cos_val;
    
    k_out.x = k_pair.x * cos_val - k_pair.y * sin_val;
    k_out.y = k_pair.x * sin_val + k_pair.y * cos_val;

    // 5. 向量化写回
    Q[idx] = q_out;
    K[idx] = k_out;
}

// CPU 参考实现
void rope_cpu(float *data, int seq_len, int head_dim, int num_heads) {
    for (int m = 0; m < seq_len; m++) {
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < head_dim / 2; i++) {
                float freq = 1.0f / powf(10000.0f, 2.0f * i / (float)head_dim);
                float angle = (float)m * freq;
                float cos_val = cosf(angle);
                float sin_val = sinf(angle);

                int idx1 = m * num_heads * head_dim + h * head_dim + 2 * i;
                int idx2 = idx1 + 1;

                float x1 = data[idx1];
                float x2 = data[idx2];

                data[idx1] = x1 * cos_val - x2 * sin_val;
                data[idx2] = x1 * sin_val + x2 * cos_val;
            }
        }
    }
}

int main() {
    printf("=== CUDA RoPE (Rotary Positional Embedding) 示例 ===\n\n");
    
    int total_elements = N * HEADS * D;
    size_t size = total_elements * sizeof(float);
    
    float *h_Q = (float*)malloc(size);
    float *h_K = (float*)malloc(size);
    float *h_Q_ref = (float*)malloc(size);
    
    srand(time(NULL));
    for (int i = 0; i < total_elements; i++) {
        h_Q[i] = (float)(rand() % 100) / 100.0f;
        h_K[i] = (float)(rand() % 100) / 100.0f;
        h_Q_ref[i] = h_Q[i]; // 备份用于 CPU 验证
    }
    
    float *d_Q, *d_K;
    cudaMalloc(&d_Q, size);
    cudaMalloc(&d_K, size);
    
    // ---------------------------------------------------------
    // 1. 运行原始 Kernel
    // ---------------------------------------------------------
    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    
    int num_pairs = total_elements / 2;
    int num_blocks = (num_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    printf("配置: Seq=%d, Heads=%d, Dim=%d\n", N, HEADS, D);
    printf("启动内核: Grid=%d, Block=%d\n", num_blocks, THREADS_PER_BLOCK);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    rope_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_Q, d_K, N, D, HEADS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms_naive = 0;
    cudaEventElapsedTime(&ms_naive, start, stop);
    printf("原始 Kernel 执行时间: %.5f 毫秒\n", ms_naive);

    // ---------------------------------------------------------
    // 2. 准备优化版 Kernel (预计算表)
    // ---------------------------------------------------------
    // 表大小: [seq_len, head_dim/2]
    int table_size = N * (D / 2);
    float2 *h_CosSin = (float2*)malloc(table_size * sizeof(float2));
    
    // CPU 预计算
    for (int m = 0; m < N; m++) {
        for (int i = 0; i < D / 2; i++) {
            float freq = 1.0f / powf(10000.0f, 2.0f * i / (float)D);
            float angle = (float)m * freq;
            h_CosSin[m * (D/2) + i].x = cosf(angle);
            h_CosSin[m * (D/2) + i].y = sinf(angle);
        }
    }
    
    float2 *d_CosSin;
    cudaMalloc(&d_CosSin, table_size * sizeof(float2));
    cudaMemcpy(d_CosSin, h_CosSin, table_size * sizeof(float2), cudaMemcpyHostToDevice);
    
    // 重置 Q 和 K
    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    // 注意：这里传入的是 float2* 指针
    rope_kernel_opt<<<num_blocks, THREADS_PER_BLOCK>>>((float2*)d_Q, (float2*)d_K, d_CosSin, N, D, HEADS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms_opt = 0;
    cudaEventElapsedTime(&ms_opt, start, stop);
    printf("优化 Kernel 执行时间: %.5f 毫秒 (加速比: %.2fx)\n", ms_opt, ms_naive / ms_opt);

    // ---------------------------------------------------------
    // 3. 验证结果 (使用优化版的结果)
    // ---------------------------------------------------------
    cudaMemcpy(h_Q, d_Q, size, cudaMemcpyDeviceToHost);
    rope_cpu(h_Q_ref, N, D, HEADS);
    
    float max_diff = 0.0f;
    for (int i = 0; i < total_elements; i++) {
        float diff = fabs(h_Q[i] - h_Q_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    
    printf("最大误差: %e\n", max_diff);
    if (max_diff < 1e-4) printf("✓ 结果正确\n");
    else printf("✗ 结果可能有误\n");
    
    free(h_Q); free(h_K); free(h_Q_ref); free(h_CosSin);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_CosSin);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
