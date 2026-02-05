/*
 * CUDA 编程教程 - 第10课：RMSNorm (均方根层归一化)
 *
 * RMSNorm 是 LayerNorm 的一种简化，广泛应用于现代大语言模型 (LLM) 中。
 * 它通过只进行缩放而不进行中心化来减少计算量。
 *
 * 编译和运行:
 * nvcc 10_rmsnorm.cu -o 10_rmsnorm -O3
 * ./10_rmsnorm
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>
#include <time.h>

#define N (1024 * 1024)  // 向量大小
#define THREADS_PER_BLOCK 256

// 内核1: 并行规约，计算平方和 (Sum of Squares)
__global__ void reduce_sum_squares(const float *input, float *block_sums, int n)
{
    __shared__ float sdata[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程计算自己负责元素的平方
    float val = (i < n) ? input[i] : 0.0f;
    sdata[tid] = val * val;
    __syncthreads();
    
    // 在共享内存中进行标准的并行规约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 线程块0将该块的部分和写入全局内存
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// 内核2: 执行 RMSNorm 的主要操作
__global__ void rms_normalize(const float *input, const float *weights, float *output, float inv_rms, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // RMSNorm 公式
        output[i] = (input[i] * inv_rms) * weights[i];
    }
}

// CPU 参考实现
void rmsnorm_cpu(const float *input, const float *weights, float *output, int n)
{
    double ss = 0.0;
    for (int i = 0; i < n; i++) {
        ss += input[i] * input[i];
    }
    ss /= n;
    ss += 1e-5f; // epsilon
    ss = 1.0 / sqrt(ss);

    for (int i = 0; i < n; i++) {
        output[i] = (input[i] * (float)ss) * weights[i];
    }
}

int main()
{
    printf("=== CUDA RMSNorm 示例 ===\n\n");
    
    // 分配主机内存
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_weights = (float*)malloc(N * sizeof(float));
    float *h_output_gpu = (float*)malloc(N * sizeof(float));
    float *h_output_cpu = (float*)malloc(N * sizeof(float));
    
    // 初始化数据
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 100) / 50.0f - 1.0f; // -1.0 to 1.0
        h_weights[i] = 1.0f; // 简单起见，权重设为1
    }
    
    // 分配设备内存
    float *d_input, *d_weights, *d_output, *d_block_sums;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_weights, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));
    
    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaMalloc((void**)&d_block_sums, num_blocks * sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // --- GPU 计算 ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    // 步骤1: 计算每个线程块的平方和
    reduce_sum_squares<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_block_sums, N);
    
    // 在 CPU 上完成最终的求和（对于少量块的部分和，这足够快）
    float *h_block_sums = (float*)malloc(num_blocks * sizeof(float));
    cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_sum_sq = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        total_sum_sq += h_block_sums[i];
    }
    
    // 计算 RMS 的倒数
    float rms = sqrtf(total_sum_sq / N + 1e-5f); // 加上 epsilon
    float inv_rms = 1.0f / rms;

    // 步骤2: 执行归一化和缩放
    rms_normalize<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_weights, d_output, inv_rms, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU 执行时间: %.5f 毫秒\n", milliseconds);

    // 复制结果回主机
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // --- CPU 参考计算 ---
    rmsnorm_cpu(h_input, h_weights, h_output_cpu, N);

    // --- 验证结果 ---
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_output_gpu[i] - h_output_cpu[i]) > 1e-4) {
            printf("错误在索引 %d: GPU=%.6f vs CPU=%.6f\n", i, h_output_gpu[i], h_output_cpu[i]);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("✓ 结果正确\n");
    } else {
        printf("✗ 结果有误\n");
    }

    // 清理
    free(h_input);
    free(h_weights);
    free(h_output_gpu);
    free(h_output_cpu);
    free(h_block_sums);
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    cudaFree(d_block_sums);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
