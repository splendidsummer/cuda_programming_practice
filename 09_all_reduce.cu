/*
 * CUDA 编程教程 - 第9课：All Reduce（全局归约）操作
 * 
 * 本示例展示如何实现 All Reduce 操作：
 * 1. 所有线程块执行 reduce 操作
 * 2. 将结果广播给所有线程块
 * 3. 使用多轮内核调用实现全局归约
 * 4. 使用原子操作实现全局归约
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024 * 1024  // 数组大小
#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS 1024

// 线程块内归约
__global__ void reduce_block(float *input, float *block_output, int n)
{
    __shared__ float sdata[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据到共享内存
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // 并行归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 写入线程块结果
    if (tid == 0) {
        block_output[blockIdx.x] = sdata[0];
    }
}

// 将归约结果写入所有位置（All Reduce 的广播阶段）
__global__ void broadcast_result(float *output, float result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = result;
    }
}

// 使用原子操作的 All Reduce（简单但可能较慢）
__global__ void all_reduce_atomic(float *input, float *output, int n)
{
    __shared__ float sdata[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 线程块内归约
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 使用原子操作累加到全局结果
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
    
    // 同步所有线程块（需要多次同步）
    __syncthreads();
    
    // 读取全局结果并写入输出
    if (i < n) {
        // 注意：这里需要等待所有线程块完成归约
        // 在实际应用中，可能需要多轮内核调用
        output[i] = *output;  // 简化的实现
    }
}

// 多阶段 All Reduce：第一阶段 - 线程块内归约
__global__ void all_reduce_stage1(float *input, float *block_output, int n)
{
    __shared__ float sdata[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_output[blockIdx.x] = sdata[0];
    }
}

// 多阶段 All Reduce：第二阶段 - 递归归约线程块结果
__global__ void all_reduce_stage2(float *block_input, float *block_output, int num_blocks)
{
    __shared__ float sdata[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < num_blocks) ? block_input[i] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_output[blockIdx.x] = sdata[0];
    }
}

// 多阶段 All Reduce：第三阶段 - 广播结果
__global__ void all_reduce_stage3(float *output, float result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = result;
    }
}

// CPU 参考实现
float all_reduce_cpu(float *data, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// 多阶段 All Reduce 主函数
void all_reduce_multi_stage(float *d_input, float *d_output, int n, 
                            float *h_block_sums, float *d_block1, float *d_block2)
{
    int threads_per_block = THREADS_PER_BLOCK;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // 阶段1：线程块内归约
    all_reduce_stage1<<<num_blocks, threads_per_block>>>(d_input, d_block1, n);
    cudaDeviceSynchronize();
    
    // 阶段2：递归归约线程块结果（如果需要多轮）
    int current_blocks = num_blocks;
    float *current_input = d_block1;
    float *current_output = d_block2;
    
    while (current_blocks > 1) {
        int next_blocks = (current_blocks + threads_per_block - 1) / threads_per_block;
        all_reduce_stage2<<<next_blocks, threads_per_block>>>(current_input, current_output, current_blocks);
        cudaDeviceSynchronize();
        
        // 交换输入输出
        float *temp = current_input;
        current_input = current_output;
        current_output = temp;
        current_blocks = next_blocks;
    }
    
    // 读取最终结果
    float final_result;
    cudaMemcpy(&final_result, current_input, sizeof(float), cudaMemcpyDeviceToHost);
    
    // 阶段3：广播结果到所有元素
    all_reduce_stage3<<<num_blocks, threads_per_block>>>(d_output, final_result, n);
    cudaDeviceSynchronize();
}

int main()
{
    printf("=== CUDA All Reduce（全局归约）操作示例 ===\n\n");
    
    // 分配主机内存
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    float *h_block_sums = (float*)malloc(MAX_BLOCKS * sizeof(float));
    
    // 初始化数据
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 100) / 10.0f;  // 0-10 之间的随机数
    }
    
    // 分配设备内存
    float *d_input, *d_output;
    float *d_block1, *d_block2;  // 用于多阶段归约的中间结果
    
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));
    cudaMalloc((void**)&d_block1, MAX_BLOCKS * sizeof(float));
    cudaMalloc((void**)&d_block2, MAX_BLOCKS * sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
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
    
    // ========== 方法1：多阶段 All Reduce ==========
    printf("--- 方法1：多阶段 All Reduce ---\n");
    cudaEventRecord(start);
    all_reduce_multi_stage(d_input, d_output, N, h_block_sums, d_block1, d_block2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("执行时间：%.3f 毫秒\n", milliseconds);
    
    // 复制结果到主机
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    float expected_sum = all_reduce_cpu(h_input, N);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_output[i] - expected_sum) > 1e-3) {
            correct = false;
            if (i < 10) {  // 只打印前几个错误
                printf("错误在索引 %d: %.6f vs %.6f\n", i, h_output[i], expected_sum);
            }
            break;
        }
    }
    
    if (correct) {
        printf("✓ 结果正确\n");
        printf("All Reduce 结果（所有元素都相同）：%.6f\n", h_output[0]);
        printf("期望值：%.6f\n", expected_sum);
    } else {
        printf("✗ 结果有误\n");
    }
    printf("\n");
    
    // ========== 方法2：使用原子操作（演示，不推荐用于生产） ==========
    printf("--- 方法2：使用原子操作（简化版本） ---\n");
    printf("注意：原子操作版本可能较慢，主要用于演示\n");
    
    float *d_atomic_result;
    cudaMalloc((void**)&d_atomic_result, sizeof(float));
    float zero = 0.0f;
    cudaMemcpy(d_atomic_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    
    // 第一阶段：线程块内归约并原子累加
    reduce_block<<<num_blocks, threads_per_block>>>(d_input, d_block1, N);
    cudaDeviceSynchronize();
    
    // 读取线程块结果并在 CPU 上求和（简化实现）
    cudaMemcpy(h_block_sums, d_block1, num_blocks * sizeof(float), 
               cudaMemcpyDeviceToHost);
    float atomic_sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        atomic_sum += h_block_sums[i];
    }
    
    // 广播结果
    broadcast_result<<<num_blocks, threads_per_block>>>(d_output, atomic_sum, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("执行时间：%.3f 毫秒\n", milliseconds);
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("All Reduce 结果：%.6f\n\n", h_output[0]);
    
    // 清理
    cudaFree(d_atomic_result);
    
    // 显示部分结果
    printf("--- 结果验证（前 10 个元素） ---\n");
    for (int i = 0; i < 10 && i < N; i++) {
        printf("output[%d] = %.6f\n", i, h_output[i]);
    }
    printf("（所有元素应该相同）\n\n");
    
    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block1);
    cudaFree(d_block2);
    free(h_input);
    free(h_output);
    free(h_block_sums);
    
    printf("完成！\n");
    
    return 0;
}

/*
 * 编译和运行：
 * nvcc 09_all_reduce.cu -o all_reduce -O3
 * ./all_reduce
 * 
 * 关键概念：
 * 1. All Reduce：在所有线程块上执行 reduce，然后广播结果
 * 2. 多阶段实现：
 *    - 阶段1：线程块内归约
 *    - 阶段2：递归归约线程块结果
 *    - 阶段3：广播最终结果
 * 3. 实现方式：
 *    - 多轮内核调用（推荐）
 *    - 原子操作（简单但可能较慢）
 *    - 使用 NCCL 库（多 GPU 场景）
 * 
 * 应用场景：
 * - 分布式训练中的梯度聚合
 * - 并行计算中的全局统计
 * - 多 GPU 通信
 * 
 * 性能优化：
 * - 使用共享内存减少全局内存访问
 * - 合理选择线程块大小
 * - 对于多 GPU，使用 NCCL 库
 * - 考虑使用 CUDA-aware MPI
 */

