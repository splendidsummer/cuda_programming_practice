/*
 * CUDA 编程教程 - 第16课：全连接层 (Linear) 与 Softmax 算子
 *
 * 本示例展示深度学习中最基础的两个算子的 CUDA 实现：
 * 1. 全连接层 (Fully Connected / Linear Layer): Y = X * W + b
 *    本质上是矩阵乘法 (GEMM) 加上偏置向量加法。
 * 2. Softmax 算子: 
 *    包含三个步骤：求最大值(为了数值稳定性)、计算指数和、归一化。
 *
 * 编译和运行:
 * nvcc 16_fc_softmax.cu -o 16_fc_softmax -O3
 * ./16_fc_softmax
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// --- 超参数配置 ---
#define M 4     // Batch Size (行数)
#define K 128   // Input Features (输入维度)
#define N 10    // Output Features (输出维度/分类数)

#define BLOCK_SIZE 16

// ==========================================
// 1. 全连接层 (Linear Layer)
// ==========================================

// 简单的全连接层内核: Y = X * W + b
// X: [M, K], W: [K, N], b: [N], Y: [M, N]
// 假设 W 是行主序存储 (Row-Major)
__global__ void linear_layer_kernel(const float* __restrict__ X, 
                                    const float* __restrict__ W, 
                                    const float* __restrict__ b, 
                                    float* __restrict__ Y, 
                                    int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        // 计算点积 (Dot Product)
        for (int i = 0; i < k; ++i) {
            // X[row, i] * W[i, col]
            sum += X[row * k + i] * W[i * n + col];
        }
        // 加上偏置 (Bias)
        if (b != NULL) {
            sum += b[col];
        }
        Y[row * n + col] = sum;
    }
}

// ==========================================
// 2. Softmax 算子
// ==========================================

// Softmax 内核: 对每一行进行 Softmax
// 策略: 每个线程块处理一行 (One Block Per Row)
// 假设 N (列数) 不超过 1024 (最大线程数)
__global__ void softmax_kernel(float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    int row = blockIdx.x; // 当前处理的行
    int tid = threadIdx.x;

    if (row >= rows) return;

    // 共享内存用于规约 (Reduction)
    // 假设 cols <= 1024，我们分配足够的共享内存
    // 实际应用中需要处理 cols > blockDim.x 的情况
    extern __shared__ float sdata[]; 

    // 1. 寻找最大值 (Max) - 为了数值稳定性
    // ---------------------------------------
    float local_val = (tid < cols) ? input[row * cols + tid] : -1e30f;
    sdata[tid] = local_val;
    __syncthreads();

    // 块内规约求最大值
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // 2. 计算指数并求和 (Exp & Sum)
    // ---------------------------------------
    local_val = (tid < cols) ? expf(local_val - max_val) : 0.0f;
    sdata[tid] = local_val;
    __syncthreads();

    // 块内规约求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0];
    __syncthreads();

    // 3. 归一化 (Normalize)
    // ---------------------------------------
    if (tid < cols) {
        // 重新读取局部值 (或者使用寄存器缓存)
        // 这里直接使用之前计算的 exp 值 (如果寄存器够用，local_val 还在)
        // 但为了演示清晰，我们用 sdata[tid] (注意：sdata 在规约过程中被修改了，所以不能直接用)
        // 让我们重新计算 exp，或者在规约前保存一份。
        // 简单起见，我们重新计算 exp (或者如果 cols 很小，每个线程负责写入)
        
        // 更正：上面的规约破坏了 sdata。
        // 实际上，最高效的方法是每个线程计算完 exp 后，不依赖 sdata 存所有值，
        // 而是只用 sdata 做规约。每个线程保留自己的 local_val (exp值)。
        
        output[row * cols + tid] = local_val / sum_exp;
    }
}

// ==========================================
// Host 代码
// ==========================================

void cpu_linear(float* X, float* W, float* b, float* Y, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += X[i * k + l] * W[l * n + j];
            }
            if (b) sum += b[j];
            Y[i * n + j] = sum;
        }
    }
}

void cpu_softmax(float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float max_val = -1e30f;
        for (int j = 0; j < cols; ++j) {
            if (input[i * cols + j] > max_val) max_val = input[i * cols + j];
        }
        
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float val = expf(input[i * cols + j] - max_val);
            output[i * cols + j] = val;
            sum += val;
        }
        
        for (int j = 0; j < cols; ++j) {
            output[i * cols + j] /= sum;
        }
    }
}

int main() {
    printf("=== CUDA Linear Layer & Softmax Tutorial ===\n");
    printf("Matrix Size: M=%d, K=%d, N=%d\n", M, K, N);

    size_t size_X = M * K * sizeof(float);
    size_t size_W = K * N * sizeof(float);
    size_t size_b = N * sizeof(float);
    size_t size_Y = M * N * sizeof(float);

    // 1. 初始化数据
    float *h_X = (float*)malloc(size_X);
    float *h_W = (float*)malloc(size_W);
    float *h_b = (float*)malloc(size_b);
    float *h_Y_gpu = (float*)malloc(size_Y);
    float *h_Y_cpu = (float*)malloc(size_Y);
    float *h_Softmax_gpu = (float*)malloc(size_Y);
    float *h_Softmax_cpu = (float*)malloc(size_Y);

    srand(time(NULL));
    for(int i=0; i<M*K; i++) h_X[i] = (rand() % 100) / 100.0f;
    for(int i=0; i<K*N; i++) h_W[i] = (rand() % 100) / 100.0f;
    for(int i=0; i<N; i++) h_b[i] = (rand() % 100) / 100.0f;

    // 2. 设备内存分配
    float *d_X, *d_W, *d_b, *d_Y, *d_Softmax;
    cudaMalloc(&d_X, size_X);
    cudaMalloc(&d_W, size_W);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_Y, size_Y);
    cudaMalloc(&d_Softmax, size_Y);

    cudaMemcpy(d_X, h_X, size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, size_W, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    // 3. 执行 Linear Layer
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("\nRunning Linear Layer Kernel...\n");
    linear_layer_kernel<<<grid, block>>>(d_X, d_W, d_b, d_Y, M, N, K);
    cudaDeviceSynchronize();

    // 4. 执行 Softmax
    // 假设 N <= 1024，我们用一个 Block 处理一行
    // 动态共享内存大小: N * sizeof(float)
    printf("Running Softmax Kernel...\n");
    int shared_mem_size = N * sizeof(float);
    // 确保线程数至少是 N 的下一个 2 的幂次方，以便进行规约
    int threads = 1;
    while(threads < N) threads *= 2;
    
    softmax_kernel<<<M, threads, shared_mem_size>>>(d_Y, d_Softmax, M, N);
    cudaDeviceSynchronize();

    // 5. 验证结果
    cudaMemcpy(h_Y_gpu, d_Y, size_Y, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Softmax_gpu, d_Softmax, size_Y, cudaMemcpyDeviceToHost);

    cpu_linear(h_X, h_W, h_b, h_Y_cpu, M, N, K);
    cpu_softmax(h_Y_cpu, h_Softmax_cpu, M, N);

    // 检查 Linear 误差
    float max_diff_linear = 0.0f;
    for(int i=0; i<M*N; i++) {
        float diff = fabs(h_Y_gpu[i] - h_Y_cpu[i]);
        if(diff > max_diff_linear) max_diff_linear = diff;
    }
    printf("Linear Layer Max Diff: %e\n", max_diff_linear);

    // 检查 Softmax 误差
    float max_diff_softmax = 0.0f;
    for(int i=0; i<M*N; i++) {
        float diff = fabs(h_Softmax_gpu[i] - h_Softmax_cpu[i]);
        if(diff > max_diff_softmax) max_diff_softmax = diff;
    }
    printf("Softmax Layer Max Diff: %e\n", max_diff_softmax);

    if (max_diff_linear < 1e-4 && max_diff_softmax < 1e-4) {
        printf("✓ Results Verified!\n");
    } else {
        printf("✗ Results Mismatch!\n");
    }

    // 打印部分结果
    printf("\nSample Output (Row 0):\n");
    printf("Linear (GPU): ");
    for(int i=0; i<5 && i<N; i++) printf("%.4f ", h_Y_gpu[i]);
    printf("\nSoftmax (GPU): ");
    for(int i=0; i<5 && i<N; i++) printf("%.4f ", h_Softmax_gpu[i]);
    printf("\n");

    // 清理
    free(h_X); free(h_W); free(h_b); free(h_Y_gpu); free(h_Y_cpu); free(h_Softmax_gpu); free(h_Softmax_cpu);
    cudaFree(d_X); cudaFree(d_W); cudaFree(d_b); cudaFree(d_Y); cudaFree(d_Softmax);

    return 0;
}
