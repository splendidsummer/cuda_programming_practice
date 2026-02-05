/*
 * CUDA 编程教程 - 第8课：Reduce（归约）操作
 * 
 * 本示例展示如何实现 Reduce 操作：
 * 1. 线程块内的归约（使用共享内存）
 * 2. 全局归约（跨线程块）
 * 3. 不同归约操作（求和、求最大值、求最小值）
 * 4. 优化的归约算法
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024 * 1024  // 数组大小
#define THREADS_PER_BLOCK 256

// 简单的归约：每个线程块独立归约
__global__ void reduce_block_sum(float *input, float *output, int n)
{
    __shared__ float sdata[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 从全局内存加载数据到共享内存
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // 并行归约：树形结构
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 将线程块的结果写入全局内存
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// 优化的归约：避免 bank conflicts
__global__ void reduce_optimized_sum(float *input, float *output, int n)
{
    __shared__ float sdata[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // 优化的归约：使用展开循环
    if (blockDim.x >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    
    // 使用 warp 内的洗牌指令（warp shuffle）
    if (tid < 32) {
        // 在 warp 内进行最终的归约
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
    
    // 写入结果
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// 求最大值归约
__global__ void reduce_max(float *input, float *output, int n)
{
    __shared__ float sdata[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据
    sdata[tid] = (i < n) ? input[i] : -1e30f;  // 使用很大的负数代替 -INFINITY
    __syncthreads();
    
    // 并行归约：求最大值
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // 写入结果
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// 求最小值归约
__global__ void reduce_min(float *input, float *output, int n)
{
    __shared__ float sdata[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据
    sdata[tid] = (i < n) ? input[i] : 1e30f;  // 使用很大的正数代替 INFINITY
    __syncthreads();
    
    // 并行归约：求最小值
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // 写入结果
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// 这个内核通过特定的内存访问模式，故意引发严重的银行冲突
__global__ void reduce_with_bank_conflicts(float *input, float *output, int n)
{
    // __shared__ 内存被划分为 32 个 bank
    // 地址为 addr 的数据存放在 bank (addr % 32)
    __shared__ float sdata[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 从全局内存加载数据到共享内存（和之前一样）
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // --- 产生银行冲突的归约 ---
    // 在这个循环中，同一个 Warp 内的多个线程会访问同一个 Bank
    for (int s = 1; s < blockDim.x; s *= 2) {
        // 关键点：让线程去访问以 2*s 为步长的数据
        // 当 2*s 是 32 的倍数时，就会发生冲突
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

// CPU 参考实现：求和
float reduce_sum_cpu(float *data, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// CPU 参考实现：求最大值
float reduce_max_cpu(float *data, int n)
{
    float max_val = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
}

// CPU 参考实现：求最小值
float reduce_min_cpu(float *data, int n)
{
    float min_val = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] < min_val) {
            min_val = data[i];
        }
    }
    return min_val;
}

int main()
{
    printf("=== CUDA Reduce（归约）操作示例 ===\n\n");
    
    // 分配主机内存
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_block_sums = NULL;
    float *h_output = (float*)malloc(sizeof(float));
    
    // 初始化数据
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 1000) / 100.0f;  // 0-10 之间的随机数
    }
    
    // 分配设备内存
    float *d_input, *d_output, *d_block_output;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 配置线程
    int threads_per_block = THREADS_PER_BLOCK;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    
    printf("数组大小：%d\n", N);
    printf("线程块数：%d\n", num_blocks);
    printf("每块线程数：%d\n\n", threads_per_block);
    
    // 分配线程块结果内存
    h_block_sums = (float*)malloc(num_blocks * sizeof(float));
    cudaMalloc((void**)&d_block_output, num_blocks * sizeof(float));
    
    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    // ========== 求和归约 ==========
    printf("--- 求和归约 (Sum Reduce) ---\n");
    
    // 第一阶段：每个线程块内部归约
    cudaEventRecord(start);
    reduce_block_sum<<<num_blocks, threads_per_block>>>(d_input, d_block_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("第一阶段（线程块内归约）时间：%.3f 毫秒\n", milliseconds);
    
    // 复制线程块结果到主机
    cudaMemcpy(h_block_sums, d_block_output, num_blocks * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // 第二阶段：在 CPU 上归约线程块结果（如果只有一个线程块，可以跳过）
    if (num_blocks > 1) {
        // 可以在 GPU 上再次归约，这里为了简单在 CPU 上完成
        float final_sum = 0.0f;
        for (int i = 0; i < num_blocks; i++) {
            final_sum += h_block_sums[i];
        }
        *h_output = final_sum;
    } else {
        *h_output = h_block_sums[0];
    }
    
    // CPU 参考
    float cpu_sum = reduce_sum_cpu(h_input, N);
    printf("GPU 结果：%.6f\n", *h_output);
    printf("CPU 结果：%.6f\n", cpu_sum);
    printf("误差：%.6f\n\n", fabs(*h_output - cpu_sum));
    
    // ========== 优化的求和归约 ==========
    printf("--- 优化的求和归约 ---\n");
    cudaEventRecord(start);
    reduce_optimized_sum<<<num_blocks, threads_per_block>>>(d_input, d_block_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("优化版本时间：%.3f 毫秒\n", milliseconds);
    
    cudaMemcpy(h_block_sums, d_block_output, num_blocks * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    if (num_blocks > 1) {
        float final_sum = 0.0f;
        for (int i = 0; i < num_blocks; i++) {
            final_sum += h_block_sums[i];
        }
        *h_output = final_sum;
    } else {
        *h_output = h_block_sums[0];
    }
    printf("GPU 结果：%.6f\n\n", *h_output);
    
    // ========== 求最大值归约 ==========
    printf("--- 求最大值归约 (Max Reduce) ---\n");
    reduce_max<<<num_blocks, threads_per_block>>>(d_input, d_block_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_block_sums, d_block_output, num_blocks * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    float gpu_max = h_block_sums[0];
    for (int i = 1; i < num_blocks; i++) {
        if (h_block_sums[i] > gpu_max) {
            gpu_max = h_block_sums[i];
        }
    }
    
    float cpu_max = reduce_max_cpu(h_input, N);
    printf("GPU 最大值：%.6f\n", gpu_max);
    printf("CPU 最大值：%.6f\n", cpu_max);
    printf("误差：%.6f\n\n", fabs(gpu_max - cpu_max));
    
    // ========== 求最小值归约 ==========
    printf("--- 求最小值归约 (Min Reduce) ---\n");
    reduce_min<<<num_blocks, threads_per_block>>>(d_input, d_block_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_block_sums, d_block_output, num_blocks * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    float gpu_min = h_block_sums[0];
    for (int i = 1; i < num_blocks; i++) {
        if (h_block_sums[i] < gpu_min) {
            gpu_min = h_block_sums[i];
        }
    }
    
    float cpu_min = reduce_min_cpu(h_input, N);
    printf("GPU 最小值：%.6f\n", gpu_min);
    printf("CPU 最小值：%.6f\n", cpu_min);
    printf("误差：%.6f\n\n", fabs(gpu_min - cpu_min));
    
    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block_output);
    free(h_input);
    free(h_block_sums);
    free(h_output);
    
    printf("完成！\n");
    
    return 0;
}

/*
 * 编译和运行：
 * nvcc 08_reduce.cu -o reduce -O3
 * ./reduce
 * 
 * 关键概念：
 * 1. Reduce（归约）：将多个值合并成一个值
 * 2. 两阶段归约：
 *    - 第一阶段：线程块内归约（使用共享内存）
 *    - 第二阶段：线程块间归约
 * 3. 优化技巧：
 *    - 使用共享内存减少全局内存访问
 *    - 避免 bank conflicts
 *    - 使用 warp shuffle 指令
 *    - 循环展开
 * 4. 常见归约操作：
 *    - 求和 (Sum)
 *    - 求最大值 (Max)
 *    - 求最小值 (Min)
 *    - 求乘积 (Product)
 * 
 * 性能提示：
 * - 共享内存大小影响线程块大小
 * - 合理选择线程块大小（通常是 256 或 512）
 * - 使用多轮内核调用来处理大量线程块
 */

