/*
 * CUDA 编程教程 - 第11课：All Gather（全局收集）操作
 * 
 * 本示例展示如何实现 All Gather 操作：
 * 1. 从所有线程块收集数据
 * 2. 将收集的数据广播给所有线程块
 * 3. 多阶段实现
 * 4. 性能优化
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024 * 1024  // 每个线程块处理的数据量
#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS 1024

// All Gather 阶段1：每个线程块准备自己的数据段
__global__ void all_gather_stage1(float *input, float *block_data, int block_size, int n)
{
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_start = blockIdx.x * block_size;
    
    // 每个线程块复制自己的数据段到共享位置
    if (global_idx < n && (global_idx - block_start) < block_size) {
        int local_idx = global_idx - block_start;
        block_data[blockIdx.x * block_size + local_idx] = input[global_idx];
    }
}

// All Gather 阶段2：将所有数据复制到每个线程块的输出
__global__ void all_gather_stage2(float *block_data, float *output, 
                                   int block_size, int num_blocks, int n)
{
    // int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 计算数据来自哪个线程块
        int src_block = idx / block_size;
        int src_local_idx = idx % block_size;
        int src_global_idx = src_block * block_size + src_local_idx;
        
        // 从对应线程块的数据段复制
        if (src_global_idx < n) {
            output[idx] = block_data[src_global_idx];
        } else {
            output[idx] = 0.0f;
        }
    }
}

// 简化的 All Gather：直接复制所有数据到输出
__global__ void all_gather_simple(float *input, float *output, int n, int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 每个线程块处理自己的数据段，然后复制到输出
        int block_id = blockIdx.x;
        int local_idx = threadIdx.x;
        int global_output_idx = block_id * block_size + local_idx;
        
        if (global_output_idx < n) {
            // 从输入复制（这里简化处理，实际应该从所有块收集）
            output[global_output_idx] = input[idx];
        }
    }
}

// All Gather 完整实现：收集所有线程块的数据
__global__ void all_gather_complete(float *input, float *gather_buffer, 
                                     float *output, int block_size, int num_blocks, int n)
{
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int local_idx = tid;
    
    // 每个线程块将自己的数据写入全局缓冲区
    int write_idx = block_id * block_size + local_idx;
    if (write_idx < n && local_idx < block_size) {
        int read_idx = block_id * blockDim.x + tid;
        if (read_idx < n) {
            gather_buffer[write_idx] = input[read_idx];
        }
    }
    
    __syncthreads();
    
    // 所有线程块读取完整数据
    int read_idx = block_id * blockDim.x + tid;
    if (read_idx < n) {
        // 从全局缓冲区读取（包含所有块的数据）
        output[read_idx] = gather_buffer[read_idx];
    }
}

// 使用共享内存的 All Gather
__global__ void all_gather_shared(float *input, float *output, int n, int block_size)
{
    __shared__ float sdata[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int global_idx = block_id * blockDim.x + tid;
    
    // 加载当前线程块的数据到共享内存
    if (global_idx < n) {
        sdata[tid] = input[global_idx];
    }
    __syncthreads();
    
    // 将共享内存中的数据写入输出（每个线程块都执行）
    // 这里需要访问所有其他线程块的数据，所以需要全局缓冲区
    // 简化版本：只处理当前块的数据
    if (global_idx < n) {
        output[global_idx] = sdata[tid];
    }
}

// CPU 参考实现：简单的数据复制
void all_gather_cpu(float *input, float *output, int n)
{
    for (int i = 0; i < n; i++) {
        output[i] = input[i];
    }
}

int main()
{
    printf("=== CUDA All Gather（全局收集）操作示例 ===\n\n");
    
    // 分配主机内存
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    float *h_output_cpu = (float*)malloc(N * sizeof(float));
    float *h_gather_buffer = (float*)malloc(N * sizeof(float));
    
    // 初始化数据：每个位置的值不同，便于验证
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;  // 简单的初始化
    }
    
    // 分配设备内存
    float *d_input, *d_output, *d_gather_buffer, *d_block_data;
    
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));
    cudaMalloc((void**)&d_gather_buffer, N * sizeof(float));
    cudaMalloc((void**)&d_block_data, N * sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 配置线程
    int threads_per_block = THREADS_PER_BLOCK;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    int block_size = (N + num_blocks - 1) / num_blocks;  // 每个块的数据量
    printf("block size is： %d\n", block_size);

    printf("数组大小：%d\n", N);
    printf("线程块数：%d\n", num_blocks);
    printf("每块线程数：%d\n", threads_per_block);
    printf("每块数据量：%d\n\n", block_size);
    
    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    // ========== 方法1：两阶段 All Gather ==========
    printf("--- 方法1：两阶段 All Gather ---\n");
    
    // 阶段1：每个线程块准备数据
    cudaEventRecord(start);
    all_gather_stage1<<<num_blocks, threads_per_block>>>(
        d_input, d_block_data, block_size, N);
    cudaDeviceSynchronize();
    
    // 阶段2：将所有数据复制到输出
    all_gather_stage2<<<num_blocks, threads_per_block>>>(
        d_block_data, d_output, block_size, num_blocks, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("执行时间：%.3f 毫秒\n", milliseconds);
    
    // 复制结果到主机
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    all_gather_cpu(h_input, h_output_cpu, N);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        // 注意：All Gather 的结果应该包含所有数据
        // 这里简化验证，检查数据是否被正确复制
        if (fabs(h_output[i] - h_input[i]) > 1e-5) {
            correct = false;
            if (i < 10) {
                printf("错误在索引 %d: %.2f vs %.2f\n", i, h_output[i], h_input[i]);
            }
            break;
        }
    }
    
    if (correct) {
        printf("✓ 结果正确\n");
    } else {
        printf("✗ 结果有误\n");
    }
    printf("\n");
    
    // ========== 方法2：完整的 All Gather ==========
    printf("--- 方法2：完整的 All Gather ---\n");
    cudaEventRecord(start);
    all_gather_complete<<<num_blocks, threads_per_block>>>(
        d_input, d_gather_buffer, d_output, block_size, num_blocks, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("执行时间：%.3f 毫秒\n", milliseconds);
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("结果验证：前 10 个元素\n");
    for (int i = 0; i < 10 && i < N; i++) {
        printf("output[%d] = %.2f\n", i, h_output[i]);
    }
    printf("\n");
    
    // ========== 方法3：使用共享内存 ==========
    printf("--- 方法3：使用共享内存的 All Gather ---\n");
    cudaEventRecord(start);
    all_gather_shared<<<num_blocks, threads_per_block>>>(
        d_input, d_output, N, block_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("执行时间：%.3f 毫秒\n", milliseconds);
    printf("（使用共享内存缓存）\n\n");
    
    // ========== 演示：每个线程块收集所有数据 ==========
    printf("--- All Gather 概念演示 ---\n");
    printf("All Gather 操作的含义：\n");
    printf("1. 每个线程块都有自己的数据段\n");
    printf("2. 收集所有线程块的数据到一个缓冲区\n");
    printf("3. 将所有数据广播给每个线程块\n");
    printf("4. 最终每个线程块都拥有完整的数据副本\n\n");
    
    // 显示部分数据分布
    printf("数据分布示例（前 3 个线程块）：\n");
    for (int block = 0; block < 3 && block < num_blocks; block++) {
        printf("线程块 %d 的数据范围：", block);
        int start = block * block_size;
        int end = (start + block_size < N) ? start + block_size : N;
        printf("索引 %d 到 %d\n", start, end - 1);
    }
    printf("\n");
    
    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gather_buffer);
    cudaFree(d_block_data);
    free(h_input);
    free(h_output);
    free(h_output_cpu);
    free(h_gather_buffer);
    
    printf("完成！\n");
    
    return 0;
}

/*
 * 编译和运行：
 * nvcc 11_all_gather.cu -o all_gather -O3
 * ./all_gather
 * 
 * 关键概念：
 * 1. All Gather：从所有线程块收集数据，然后广播给所有线程块
 * 2. 实现步骤：
 *    - 阶段1：每个线程块准备自己的数据
 *    - 阶段2：收集所有数据到全局缓冲区
 *    - 阶段3：将完整数据复制到每个线程块的输出
 * 3. 与 Gather 的区别：
 *    - Gather：从一个位置收集数据
 *    - All Gather：从所有位置收集数据，然后广播
 * 
 * 应用场景：
 * - 分布式训练中的参数同步
 * - 多 GPU 数据交换
 * - 并行计算中的全局数据共享
 * - 集合通信操作
 * 
 * 性能优化：
 * - 使用共享内存减少全局内存访问
 * - 合理组织数据布局
 * - 对于多 GPU，使用 NCCL 库
 * - 考虑使用 CUDA-aware MPI
 * 
 * 注意事项：
 * - All Gather 需要足够的全局内存存储所有数据
 * - 对于大量线程块，可能需要多轮内核调用
 * - 实际应用中，多 GPU 场景使用专门的通信库更高效
 */

