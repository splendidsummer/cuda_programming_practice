/*
 * reduce_bank_conflict.cu
 * 故意引发银行冲突的并行规约实现。
 *
 * 编译和运行:
 * nvcc reduce_bank_conflict.cu -o reduce_bank_conflict -O3
 * ./reduce_bank_conflict
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>
#include <time.h>

#define N (1024 * 1024 * 16) // 使用一个较大的数组以凸显性能差异
#define THREADS_PER_BLOCK 256

// 引发银行冲突的规约内核
__global__ void reduce_with_bank_conflicts(float *input, float *output, int n)
{
    __shared__ float sdata[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 从全局内存加载数据到共享内存
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // 产生银行冲突的归约
    // 当 2*s 是 32 的倍数时，Warp 内的线程会集中访问同一个 Bank
    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    
    // 将线程块的结果写入全局内存
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main()
{
    printf("--- 低效规约版本 (有银行冲突) ---\n");
    
    // 分配主机内存
    float *h_input = (float*)malloc(N * sizeof(float));
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 100) / 10.0f; // 一个 0.0 到 9.9 之间的随机数
    }
    
    // 分配设备内存
    float *d_input, *d_block_output;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 配置线程
    int threads_per_block = THREADS_PER_BLOCK;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    
    printf("数组大小: %d\n", N);
    printf("线程块数: %d\n", num_blocks);
    printf("每块线程数: %d\n\n", threads_per_block);
    
    // 分配用于存放每个块的部分和的内存
    cudaMalloc((void**)&d_block_output, num_blocks * sizeof(float));
    
    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 执行内核并计时
    cudaEventRecord(start);
    reduce_with_bank_conflicts<<<num_blocks, threads_per_block>>>(d_input, d_block_output, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("执行时间: %.5f 毫秒\n\n", milliseconds);
    
    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_block_output);
    free(h_input);
    
    return 0;
}