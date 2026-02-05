/*
 * CUDA 编程教程 - 第4课：矩阵乘法
 * 
 * 本示例展示如何实现矩阵乘法（C = A * B）：
 * 1. 简单的矩阵乘法实现
 * 2. 优化的矩阵乘法实现（使用共享内存）
 * 3. 性能对比
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define TILE_SIZE 16  // 平铺大小（用于共享内存优化）

// 简单的矩阵乘法内核（未优化）
__global__ void matrix_multiply_naive(float *A, float *B, float *C, 
                                      int M, int N, int K)
{
    // 计算当前线程处理的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        // 计算点积
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 使用共享内存优化的矩阵乘法
__global__ void matrix_multiply_tiled(float *A, float *B, float *C,
                                      int M, int N, int K)
{
    // 共享内存：每个线程块共享的内存，访问速度快
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    // 线程在线程块中的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 线程处理的行和列
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // 分块处理矩阵
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // 将 A 的一个平铺加载到共享内存
        if (row < M && (tile * TILE_SIZE + tx) < K) {
            tile_A[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        } else {
            tile_A[ty][tx] = 0.0f;
        }
        
        // 将 B 的一个平铺加载到共享内存
        if ((tile * TILE_SIZE + ty) < K && col < N) {
            tile_B[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            tile_B[ty][tx] = 0.0f;
        }
        
        // 同步：等待所有线程完成数据加载
        __syncthreads();
        
        // 计算点积
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }
        
        // 同步：等待所有线程完成计算
        __syncthreads();
    }
    
    // 写入结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// CPU 参考实现
void matrix_multiply_cpu(float *A, float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 验证结果
bool verify_result(float *C1, float *C2, int size, float epsilon = 1e-3)
{
    for (int i = 0; i < size; i++) {
        if (fabs(C1[i] - C2[i]) > epsilon) {
            printf("不匹配在索引 %d: %.6f vs %.6f\n", i, C1[i], C2[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    printf("=== CUDA 矩阵乘法示例 ===\n\n");
    
    // 矩阵维度
    int M = 512;  // A 的行数
    int K = 512;  // A 的列数，B 的行数
    int N = 512;  // B 的列数
    // 结果矩阵 C 的维度：M x N
    
    printf("矩阵维度：A(%d x %d) * B(%d x %d) = C(%d x %d)\n\n", 
           M, K, K, N, M, N);
    
    // 分配主机内存
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_naive = (float*)malloc(size_C);
    float *h_C_tiled = (float*)malloc(size_C);
    float *h_C_cpu = (float*)malloc(size_C);
    
    // 初始化矩阵
    for (int i = 0; i < M * K; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);
    
    // 复制数据到设备
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // 配置线程块
    dim3 block_size(16, 16);
    dim3 grid_size_naive((N + block_size.x - 1) / block_size.x,
                         (M + block_size.y - 1) / block_size.y);
    dim3 grid_size_tiled((N + TILE_SIZE - 1) / TILE_SIZE,
                         (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    // ========== 未优化的矩阵乘法 ==========
    printf("--- 未优化的矩阵乘法 ---\n");
    cudaEventRecord(start);
    matrix_multiply_naive<<<grid_size_naive, block_size>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("执行时间：%.3f 毫秒\n", milliseconds);
    cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // ========== 使用共享内存优化的矩阵乘法 ==========
    printf("\n--- 使用共享内存优化的矩阵乘法 ---\n");
    cudaEventRecord(start);
    matrix_multiply_tiled<<<grid_size_tiled, dim3(TILE_SIZE, TILE_SIZE)>>>(
        d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("执行时间：%.3f 毫秒\n", milliseconds);
    cudaMemcpy(h_C_tiled, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // ========== CPU 参考实现 ==========
    printf("\n--- CPU 参考实现 ---\n");
    clock_t cpu_start = clock();
    matrix_multiply_cpu(h_A, h_B, h_C_cpu, M, N, K);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000;
    printf("执行时间：%.3f 毫秒\n", cpu_time);
    
    // 验证结果
    printf("\n--- 验证结果 ---\n");
    bool naive_correct = verify_result(h_C_naive, h_C_cpu, M * N);
    bool tiled_correct = verify_result(h_C_tiled, h_C_cpu, M * N);
    
    if (naive_correct) {
        printf("✓ 未优化版本结果正确\n");
    } else {
        printf("✗ 未优化版本结果有误\n");
    }
    
    if (tiled_correct) {
        printf("✓ 优化版本结果正确\n");
    } else {
        printf("✗ 优化版本结果有误\n");
    }
    
    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);
    free(h_C_cpu);
    
    printf("\n完成！\n");
    
    return 0;
}

/*
 * 编译和运行：
 * nvcc 04_matrix_multiply.cu -o matrix_multiply
 * ./matrix_multiply
 * 
 * 关键概念：
 * 1. 共享内存 (__shared__): 线程块内共享的快速内存
 * 2. __syncthreads(): 线程块内同步，确保所有线程完成某个操作
 * 3. 平铺 (Tiling): 将大矩阵分成小块处理，提高缓存利用率
 * 4. 性能优化：共享内存比全局内存快得多
 * 
 * 性能提示：
 * - 使用共享内存可以减少全局内存访问
 * - 合理选择 TILE_SIZE（通常是 16 或 32）
 * - 使用 cudaEvent 可以精确测量 GPU 执行时间
 */

