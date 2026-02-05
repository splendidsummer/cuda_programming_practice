/*
 * CUDA 编程教程 - 第13课：SwiGLU 激活函数
 *
 * SwiGLU (Swish Gated Linear Unit) 是 LLaMA 等现代 LLM 使用的激活函数。
 * 它通常替代了传统的 ReLU 或 GeLU。
 *
 * 数学定义:
 * SwiGLU(x) = Swish_beta(xW) * (xV)
 * 其中 Swish_beta(x) = x * Sigmoid(beta * x)
 * 在 LLaMA 中，通常 beta=1，即 SiLU (Sigmoid Linear Unit)。
 *
 * 输入:
 * 通常有两个输入张量 (来自两个不同的线性层投影):
 * 1. Gate (xW)
 * 2. Up (xV)
 *
 * 操作:
 * Output = (Gate * Sigmoid(Gate)) * Up
 * 这是一个逐元素 (Element-wise) 操作。
 *
 * 编译和运行:
 * nvcc 13_swiglu.cu -o 13_swiglu -O3
 * ./13_swiglu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>

#define N 1024 * 1024 // 元素总数 (例如 batch * seq * hidden)
#define THREADS_PER_BLOCK 256

// SwiGLU Kernel
// 使用 float4 向量化加载来优化内存带宽
__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 向量化处理: 每个线程处理 4 个 float
    // 注意: 确保 n 是 4 的倍数，且指针是对齐的
    int vec_idx = idx * 4;
    
    if (vec_idx < n) {
        // 使用 reinterpret_cast 加载 float4
        float4 g_val = reinterpret_cast<const float4*>(gate)[idx];
        float4 u_val = reinterpret_cast<const float4*>(up)[idx];
        float4 res;

        // 处理第 1 个元素
        float val = g_val.x;
        float silu = val / (1.0f + expf(-val));
        res.x = silu * u_val.x;

        // 处理第 2 个元素
        val = g_val.y;
        silu = val / (1.0f + expf(-val));
        res.y = silu * u_val.y;

        // 处理第 3 个元素
        val = g_val.z;
        silu = val / (1.0f + expf(-val));
        res.z = silu * u_val.z;

        // 处理第 4 个元素
        val = g_val.w;
        silu = val / (1.0f + expf(-val));
        res.w = silu * u_val.w;

        // 写回结果
        reinterpret_cast<float4*>(out)[idx] = res;
    }
}

// CPU 参考实现
void swiglu_cpu(const float* gate, const float* up, float* out, int n) {
    for (int i = 0; i < n; i++) {
        float val = gate[i];
        float silu = val / (1.0f + expf(-val));
        out[i] = silu * up[i];
    }
}

int main() {
    printf("=== CUDA SwiGLU 激活函数示例 (向量化加载) ===\n\n");
    
    size_t size = N * sizeof(float);
    
    float *h_gate = (float*)malloc(size);
    float *h_up = (float*)malloc(size);
    float *h_out = (float*)malloc(size);
    float *h_out_ref = (float*)malloc(size);
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_gate[i] = (float)((rand() % 200) - 100) / 100.0f; // -1.0 到 1.0
        h_up[i] = (float)((rand() % 200) - 100) / 100.0f;
    }
    
    float *d_gate, *d_up, *d_out;
    cudaMalloc(&d_gate, size);
    cudaMalloc(&d_up, size);
    cudaMalloc(&d_out, size);
    
    cudaMemcpy(d_gate, h_gate, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, h_up, size, cudaMemcpyHostToDevice);
    
    // 计算 Grid 大小
    // 因为使用了 float4，每个线程处理 4 个元素，所以总线程数是 N/4
    int num_threads_needed = N / 4;
    int num_blocks = (num_threads_needed + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    printf("数据量: %d float\n", N);
    printf("启动内核: Grid=%d, Block=%d (每个线程处理 4 个元素)\n", num_blocks, THREADS_PER_BLOCK);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    swiglu_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_gate, d_up, d_out, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU 执行时间: %.5f 毫秒\n", milliseconds);
    
    // 计算带宽
    // 读取 2 * N * 4 bytes, 写入 1 * N * 4 bytes -> 总共 12 * N bytes
    double bandwidth = (3.0 * N * sizeof(float)) / (milliseconds / 1000.0) / 1e9;
    printf("有效带宽: %.2f GB/s\n", bandwidth);
    
    // 验证结果
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    swiglu_cpu(h_gate, h_up, h_out_ref, N);
    
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabs(h_out[i] - h_out_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    
    printf("最大误差: %e\n", max_diff);
    if (max_diff < 1e-4) printf("✓ 结果正确\n");
    else printf("✗ 结果可能有误\n");
    
    free(h_gate); free(h_up); free(h_out); free(h_out_ref);
    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
