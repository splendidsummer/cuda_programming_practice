/*
 * CUDA 编程教程 - 第5课：共享内存详解
 * 
 * 本示例深入讲解共享内存的使用：
 * 1. 共享内存的基础使用
 * 2. 减少全局内存访问
 * 3. 并行归约（Reduction）示例
 * 4. 共享内存的同步
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024 * 1024  // 数组大小
#define THREADS_PER_BLOCK 256

// 未优化的向量求和（直接使用全局内存）
__global__ void reduce_naive(float *input, float *output, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程处理一个元素，累加到输出
    if (tid < n) {
        atomicAdd(output, input[tid]);  // 原子操作，较慢
    }
}

// 使用共享内存的并行归约（优化版本）
__global__ void reduce_shared_memory(float *input, float *output, int n)
{
    // 共享内存：线程块内共享
    __shared__ float sdata[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 从全局内存加载数据到共享内存
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    
    // 同步：确保所有数据都加载到共享内存
    __syncthreads();
    
    // 并行归约：树形结构
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // 每次迭代后同步
    }
    
    // 将线程块的结果写入全局内存
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// 向量点积示例（使用共享内存）
__global__ void dot_product(float *a, float *b, float *result, int n)
{
    __shared__ float cache[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cache_index = threadIdx.x;
    
    float temp = 0.0f;
    
    // 计算点积的一部分
    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;  // 跨步访问
    }
    
    // 将结果存入共享内存
    cache[cache_index] = temp;
    
    // 同步
    __syncthreads();
    
    // 归约：将所有部分和相加
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cache_index < i) {
            cache[cache_index] += cache[cache_index + i];
        }
        __syncthreads();
        i /= 2;
    }
    
    // 将线程块的结果写入全局内存
    if (cache_index == 0) {
        atomicAdd(result, cache[0]);
    }
}

// CPU 参考实现：向量求和
float sum_cpu(float *data, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// CPU 参考实现：向量点积
float dot_product_cpu(float *a, float *b, int n)
{
    float result = 0.0f;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

int main()
{
    printf("=== CUDA 共享内存详解 ===\n\n");
    
    // 分配主机内存
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
        h_a[i] = (float)(i % 10);
        h_b[i] = (float)(i % 10);
    }
    
    // 分配设备内存
    float *d_input, *d_a, *d_b;
    float *d_output_naive, *d_output_shared, *d_dot_result;
    
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_output_naive, sizeof(float));
    cudaMalloc((void**)&d_output_shared, sizeof(float));
    cudaMalloc((void**)&d_dot_result, sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 初始化输出
    float h_output_naive = 0.0f;
    float h_output_shared = 0.0f;
    float h_dot_result = 0.0f;
    
    cudaMemcpy(d_output_naive, &h_output_naive, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_shared, &h_output_shared, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dot_result, &h_dot_result, sizeof(float), cudaMemcpyHostToDevice);
    
    // 配置线程
    int threads_per_block = THREADS_PER_BLOCK;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    
    printf("数组大小：%d\n", N);
    printf("线程块数：%d\n", num_blocks);
    printf("每块线程数：%d\n\n", threads_per_block);
    
    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    // ========== 未优化的向量求和 ==========
    printf("--- 未优化的向量求和 ---\n");
    cudaEventRecord(start);
    reduce_naive<<<num_blocks, threads_per_block>>>(d_input, d_output_naive, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("执行时间：%.3f 毫秒\n", milliseconds);
    cudaMemcpy(&h_output_naive, d_output_naive, sizeof(float), cudaMemcpyDeviceToHost);
    printf("结果：%.2f\n\n", h_output_naive);
    
    // ========== 使用共享内存的向量求和 ==========
    printf("--- 使用共享内存的向量求和 ---\n");
    cudaMemcpy(d_output_shared, &h_output_shared, sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    reduce_shared_memory<<<num_blocks, threads_per_block>>>(d_input, d_output_shared, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("执行时间：%.3f 毫秒\n", milliseconds);
    cudaMemcpy(&h_output_shared, d_output_shared, sizeof(float), cudaMemcpyDeviceToHost);
    printf("结果：%.2f\n\n", h_output_shared);
    
    // ========== 向量点积 ==========
    printf("--- 向量点积 ---\n");
    cudaMemcpy(d_dot_result, &h_dot_result, sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    dot_product<<<num_blocks, threads_per_block>>>(d_a, d_b, d_dot_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("执行时间：%.3f 毫秒\n", milliseconds);
    cudaMemcpy(&h_dot_result, d_dot_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU 结果：%.2f\n", h_dot_result);
    
    // CPU 参考
    float cpu_sum = sum_cpu(h_input, N);
    float cpu_dot = dot_product_cpu(h_a, h_b, N);
    printf("CPU 求和结果：%.2f\n", cpu_sum);
    printf("CPU 点积结果：%.2f\n\n", cpu_dot);
    
    // 验证结果
    printf("--- 验证结果 ---\n");
    if (fabs(h_output_shared - cpu_sum) < 1e-3) {
        printf("✓ 向量求和结果正确\n");
    } else {
        printf("✗ 向量求和结果有误\n");
    }
    
    if (fabs(h_dot_result - cpu_dot) < 1e-3) {
        printf("✓ 向量点积结果正确\n");
    } else {
        printf("✗ 向量点积结果有误\n");
    }
    
    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_output_naive);
    cudaFree(d_output_shared);
    cudaFree(d_dot_result);
    free(h_input);
    free(h_a);
    free(h_b);
    
    printf("\n完成！\n");
    
    return 0;
}

/*
 * 编译和运行：
 * nvcc 05_shared_memory.cu -o shared_memory
 * ./shared_memory
 * 
 * 关键概念：
 * 1. __shared__: 共享内存修饰符，线程块内共享
 * 2. __syncthreads(): 线程块内同步，等待所有线程到达
 * 3. 共享内存特点：
 *    - 比全局内存快得多（延迟低，带宽高）
 *    - 线程块内共享
 *    - 容量有限（通常 48KB 或 16KB）
 * 4. 并行归约：将多个值合并成一个值的高效方法
 * 5. 原子操作 (atomicAdd): 线程安全的加法操作，但较慢
 * 
 * 性能优化提示：
 * - 使用共享内存减少全局内存访问
 * - 合理使用 __syncthreads() 确保数据一致性
 * - 避免共享内存的 bank conflicts
 * - 优先使用共享内存进行线程块内的数据共享
 */

