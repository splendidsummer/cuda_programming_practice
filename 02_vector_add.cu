/*
 * CUDA 编程教程 - 第2课：向量加法
 * 
 * 本示例展示如何：
 * 1. 在 GPU 上分配内存
 * 2. 将数据从 CPU 复制到 GPU
 * 3. 在 GPU 上执行并行计算
 * 4. 将结果从 GPU 复制回 CPU
 * 5. 释放 GPU 内存
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>

#define N 1024  // 向量大小

// CUDA 内核：向量加法
// 每个线程处理一个元素
__global__ void vector_add(float *a, float *b, float *c, int n)
{
    // 计算当前线程要处理的元素索引
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 确保索引不超出数组范围
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main()
{
    printf("=== CUDA 向量加法示例 ===\n\n");
    
    // 在主机（CPU）上分配内存
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c = (float*)malloc(N * sizeof(float));
    
    // 初始化向量
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // 在设备（GPU）上分配内存
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, N * sizeof(float));
    
    // 检查内存分配是否成功
    if (d_a == NULL || d_b == NULL || d_c == NULL) {
        printf("GPU 内存分配失败！\n");
        return 1;
    }
    
    printf("内存分配成功\n");
    
    // 将数据从主机复制到设备
    // cudaMemcpyHostToDevice: 从 CPU 复制到 GPU
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("数据已复制到 GPU\n");
    
    // 配置内核启动参数
    int threads_per_block = 256;  // 每个线程块的线程数
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;  // 向上取整
    
    printf("启动内核：%d 个线程块，每个块 %d 个线程\n", 
           num_blocks, threads_per_block);
    
    // 启动内核
    vector_add<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    
    // 等待 GPU 完成
    cudaDeviceSynchronize();
    
    // 检查错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA 错误: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    // 将结果从设备复制回主机
    // cudaMemcpyDeviceToHost: 从 GPU 复制到 CPU
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("结果已复制回 CPU\n\n");
    
    // 验证结果（检查前 10 个元素）
    printf("验证结果（前 10 个元素）：\n");
    bool correct = true;
    for (int i = 0; i < 10 && i < N; i++) {
        float expected = h_a[i] + h_b[i];
        printf("c[%d] = %.2f (期望: %.2f)\n", i, h_c[i], expected);
        if (fabs(h_c[i] - expected) > 1e-5) {
            correct = false;
        }
    }
    
    if (correct) {
        printf("\n✓ 计算结果正确！\n");
    } else {
        printf("\n✗ 计算结果有误！\n");
    }
    
    // 释放 GPU 内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // 释放 CPU 内存
    free(h_a);
    free(h_b);
    free(h_c);
    
    printf("\n内存已释放\n");
    
    return 0;
}

/*
 * 编译和运行：
 * nvcc 02_vector_add.cu -o vector_add
 * ./vector_add
 * 
 * 关键概念：
 * 1. cudaMalloc: 在 GPU 上分配内存
 * 2. cudaMemcpy: 在主机和设备之间复制数据
 *    - cudaMemcpyHostToDevice: CPU -> GPU
 *    - cudaMemcpyDeviceToHost: GPU -> CPU
 * 3. cudaFree: 释放 GPU 内存
 * 4. 内存管理：所有 GPU 内存分配都必须显式释放
 * 5. 索引计算：blockIdx.x * blockDim.x + threadIdx.x
 */

