/*
 * CUDA 编程教程 - 第10课：Gather（收集）操作
 * 
 * 本示例展示如何实现 Gather 操作：
 * 1. 从多个位置收集数据到一个位置
 * 2. 使用索引数组指定收集位置
 * 3. 不同的 Gather 模式
 * 4. 性能优化技巧
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024 * 1024  // 源数组大小
#define GATHER_SIZE 1024  // 收集的数据量
#define THREADS_PER_BLOCK 256

// 简单的 Gather 操作：从源数组收集数据到目标数组
__global__ void gather_simple(float *src, float *dst, int *indices, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 从源数组的指定索引位置收集数据
        int src_idx = indices[idx];
        dst[idx] = src[src_idx];
    }
}

// Gather 操作：带边界检查
__global__ void gather_with_bounds(float *src, float *dst, int *indices, 
                                    int n, int src_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int src_idx = indices[idx];
        // 边界检查
        if (src_idx >= 0 && src_idx < src_size) {
            dst[idx] = src[src_idx];
        } else {
            dst[idx] = 0.0f;  // 越界时使用默认值
        }
    }
}

// 优化的 Gather：使用共享内存缓存索引
__global__ void gather_optimized(float *src, float *dst, int *indices, int n)
{
    __shared__ int s_indices[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 将索引加载到共享内存
    if (idx < n) {
        s_indices[tid] = indices[idx];
    }
    __syncthreads();
    
    // 从源数组收集数据
    if (idx < n) {
        dst[idx] = src[s_indices[tid]];
    }
}

// 优化：使用只读缓存 (__ldg) 加速随机访问
// const __restrict__ 告诉编译器这些指针是只读且不重叠的
__global__ void gather_readonly(const float * __restrict__ src, float *dst, const int * __restrict__ indices, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 1. 加载索引 (也可以用 __ldg)
        int src_idx = __ldg(&indices[idx]);
        
        // 2. 使用 __ldg() 强制通过只读数据缓存 (Read-Only Data Cache) 加载数据
        // 这通常比标准全局内存加载更能容忍非合并访问
        dst[idx] = __ldg(&src[src_idx]);
    }
}

// 分散-收集（Scatter-Gather）模式
__global__ void scatter_gather(float *src, float *dst, int *indices, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int src_idx = indices[idx];
        // 可以在这里进行一些计算
        dst[idx] = src[src_idx] * 2.0f;  // 示例：乘以2
    }
}

// 多维 Gather：从二维数组收集数据
__global__ void gather_2d(float *src, float *dst, int *row_indices, 
                          int *col_indices, int width, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int row = row_indices[idx];
        int col = col_indices[idx];
        int src_idx = row * width + col;
        dst[idx] = src[src_idx];
    }
}

// CPU 参考实现
void gather_cpu(float *src, float *dst, int *indices, int n)
{
    for (int i = 0; i < n; i++) {
        dst[i] = src[indices[i]];
    }
}

// 生成随机索引
void generate_indices(int *indices, int n, int src_size)
{
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        indices[i] = rand() % src_size;
    }
}

int main()
{
    printf("=== CUDA Gather（收集）操作示例 ===\n\n");
    
    // 分配主机内存
    float *h_src = (float*)malloc(N * sizeof(float));
    float *h_dst = (float*)malloc(GATHER_SIZE * sizeof(float));
    float *h_dst_cpu = (float*)malloc(GATHER_SIZE * sizeof(float));
    int *h_indices = (int*)malloc(GATHER_SIZE * sizeof(int));
    
    // 初始化源数组
    for (int i = 0; i < N; i++) {
        h_src[i] = (float)i;  // 简单的初始化
    }
    
    // 生成随机索引
    generate_indices(h_indices, GATHER_SIZE, N);
    
    printf("源数组大小：%d\n", N);
    printf("收集数据量：%d\n", GATHER_SIZE);
    printf("索引范围：0 到 %d\n\n", N - 1);
    
    // 分配设备内存
    float *d_src, *d_dst;
    int *d_indices;
    
    cudaMalloc((void**)&d_src, N * sizeof(float));
    cudaMalloc((void**)&d_dst, GATHER_SIZE * sizeof(float));
    cudaMalloc((void**)&d_indices, GATHER_SIZE * sizeof(int));
    
    // 复制数据到设备
    cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, GATHER_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    
    // 配置线程
    int threads_per_block = THREADS_PER_BLOCK;
    int num_blocks = (GATHER_SIZE + threads_per_block - 1) / threads_per_block;
    
    printf("线程块数：%d\n", num_blocks);
    printf("每块线程数：%d\n\n", threads_per_block);
    
    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    // ========== 简单的 Gather ==========
    printf("--- 简单的 Gather 操作 ---\n");
    cudaEventRecord(start);
    gather_simple<<<num_blocks, threads_per_block>>>(d_src, d_dst, d_indices, GATHER_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("执行时间：%.3f 毫秒\n", milliseconds);
    
    // 复制结果到主机
    cudaMemcpy(h_dst, d_dst, GATHER_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    // CPU 参考
    gather_cpu(h_src, h_dst_cpu, h_indices, GATHER_SIZE);
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < GATHER_SIZE; i++) {
        if (fabs(h_dst[i] - h_dst_cpu[i]) > 1e-5) {
            correct = false;
            if (i < 10) {
                printf("错误在索引 %d: %.2f vs %.2f\n", i, h_dst[i], h_dst_cpu[i]);
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
    
    // ========== 带边界检查的 Gather ==========
    printf("--- 带边界检查的 Gather ---\n");
    cudaEventRecord(start);
    gather_with_bounds<<<num_blocks, threads_per_block>>>(
        d_src, d_dst, d_indices, GATHER_SIZE, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("执行时间：%.3f 毫秒\n", milliseconds);
    printf("（包含边界检查，可能稍慢）\n\n");
    
    // ========== 优化的 Gather ==========
    printf("--- 优化的 Gather（使用共享内存） ---\n");
    cudaEventRecord(start);
    gather_optimized<<<num_blocks, threads_per_block>>>(d_src, d_dst, d_indices, GATHER_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("执行时间：%.3f 毫秒\n", milliseconds);
    printf("（使用共享内存缓存索引）\n\n");
    
    // ========== 优化的 Gather (Read-Only Cache) ==========
    printf("--- 优化的 Gather（使用只读缓存 __ldg） ---\n");
    cudaEventRecord(start);
    gather_readonly<<<num_blocks, threads_per_block>>>(d_src, d_dst, d_indices, GATHER_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("执行时间：%.3f 毫秒\n", milliseconds);
    printf("（利用纹理/只读缓存优化随机访问）\n\n");

    // ========== 分散-收集模式 ==========
    printf("--- 分散-收集模式 ---\n");
    cudaEventRecord(start);
    scatter_gather<<<num_blocks, threads_per_block>>>(d_src, d_dst, d_indices, GATHER_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("执行时间：%.3f 毫秒\n", milliseconds);
    
    cudaMemcpy(h_dst, d_dst, GATHER_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    printf("结果（前 10 个元素，已乘以2）：\n");
    for (int i = 0; i < 10 && i < GATHER_SIZE; i++) {
        printf("dst[%d] = %.2f (从 src[%d] = %.2f 收集)\n", 
               i, h_dst[i], h_indices[i], h_src[h_indices[i]]);
    }
    printf("\n");
    
    // ========== 二维 Gather ==========
    printf("--- 二维 Gather 示例 ---\n");
    int width = 100;
    int height = 100;
    int gather_2d_size = 50;
    
    float *h_src_2d = (float*)malloc(width * height * sizeof(float));
    float *h_dst_2d = (float*)malloc(gather_2d_size * sizeof(float));
    int *h_row_indices = (int*)malloc(gather_2d_size * sizeof(int));
    int *h_col_indices = (int*)malloc(gather_2d_size * sizeof(int));
    
    // 初始化二维数组
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_src_2d[i * width + j] = (float)(i * width + j);
        }
    }
    
    // 生成随机行列索引
    for (int i = 0; i < gather_2d_size; i++) {
        h_row_indices[i] = rand() % height;
        h_col_indices[i] = rand() % width;
    }
    
    float *d_src_2d, *d_dst_2d;
    int *d_row_indices, *d_col_indices;
    
    cudaMalloc((void**)&d_src_2d, width * height * sizeof(float));
    cudaMalloc((void**)&d_dst_2d, gather_2d_size * sizeof(float));
    cudaMalloc((void**)&d_row_indices, gather_2d_size * sizeof(int));
    cudaMalloc((void**)&d_col_indices, gather_2d_size * sizeof(int));
    
    cudaMemcpy(d_src_2d, h_src_2d, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_indices, h_row_indices, gather_2d_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, h_col_indices, gather_2d_size * sizeof(int), cudaMemcpyHostToDevice);
    
    int num_blocks_2d = (gather_2d_size + threads_per_block - 1) / threads_per_block;
    gather_2d<<<num_blocks_2d, threads_per_block>>>(
        d_src_2d, d_dst_2d, d_row_indices, d_col_indices, width, gather_2d_size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_dst_2d, d_dst_2d, gather_2d_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("二维数组大小：%d x %d\n", height, width);
    printf("收集数据量：%d\n", gather_2d_size);
    printf("结果（前 5 个元素）：\n");
    for (int i = 0; i < 5 && i < gather_2d_size; i++) {
        int row = h_row_indices[i];
        int col = h_col_indices[i];
        printf("dst[%d] = %.2f (从 [%d,%d] 收集)\n", 
               i, h_dst_2d[i], row, col);
    }
    printf("\n");
    
    // 清理
    cudaFree(d_src_2d);
    cudaFree(d_dst_2d);
    cudaFree(d_row_indices);
    cudaFree(d_col_indices);
    free(h_src_2d);
    free(h_dst_2d);
    free(h_row_indices);
    free(h_col_indices);
    
    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_indices);
    free(h_src);
    free(h_dst);
    free(h_dst_cpu);
    free(h_indices);
    
    printf("完成！\n");
    
    return 0;
}

/*
 * 编译和运行：
 * nvcc 10_gather.cu -o gather -O3
 * ./gather
 * 
 * 关键概念：
 * 1. Gather：从多个位置收集数据到一个位置
 * 2. 索引数组：指定从源数组的哪些位置收集数据
 * 3. 边界检查：防止访问越界
 * 4. 优化技巧：
 *    - 使用共享内存缓存索引
 *    - 合并内存访问
 *    - 避免随机访问模式
 * 
 * 应用场景：
 * - 稀疏矩阵操作
 * - 图算法中的邻接节点访问
 * - 数据重组和重排
 * - 索引选择操作
 * 
 * 性能考虑：
 * - Gather 操作的内存访问可能是非连续的
 * - 随机访问模式可能导致缓存未命中
 * - 使用共享内存可以减少全局内存访问
 * - 考虑使用纹理内存加速随机访问
 */

