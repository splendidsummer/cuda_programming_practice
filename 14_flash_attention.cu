/*
 * CUDA 编程教程 - 第11课：FlashAttention (简化版)
 *
 * FlashAttention 是一种高效的注意力机制算法，通过分块计算 (Tiling) 和
 * 在线 Softmax (Online Softmax) 技术，显著减少了对全局内存 (HBM) 的访问。
 *
 * 本示例实现了一个简化的 FlashAttention 内核，用于教学目的：
 * 1. 每个线程处理 Q 的一行（一个 Query）。
 * 2. 线程块协同加载 K 和 V 的分块到共享内存。
 * 3. 使用在线 Softmax 算法实时更新结果，无需存储巨大的注意力矩阵。
 *
 * 编译和运行:
 * nvcc 11_flash_attention.cu -o 11_flash_attention -O3
 * ./11_flash_attention
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024      // 序列长度 (Sequence Length)
#define D 64        // 维度 (Head Dimension)
#define Bc 32       // K 和 V 的分块大小 (Block size for K/V)
#define Br 128      // Q 的分块大小 (Block size for Q) = 线程块大小， num_threads_per_block

// 简单的 CPU 参考实现：标准 Attention
void attention_cpu(float *Q, float *K, float *V, float *O, int n, int d) {
    float *S = (float*)malloc(n * n * sizeof(float)); // 巨大的注意力矩阵
    float *P = (float*)malloc(n * n * sizeof(float)); // Softmax 后的概率矩阵

    // 1. 计算 S = Q * K^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += Q[i * d + k] * K[j * d + k];
            }
            S[i * n + j] = sum / sqrtf((float)d); // 缩放点积
        }
    }

    // 2. 计算 Softmax(S) -> P
    for (int i = 0; i < n; i++) {
        float max_val = -1e30f;
        for (int j = 0; j < n; j++) {
            if (S[i * n + j] > max_val) max_val = S[i * n + j];
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < n; j++) {
            P[i * n + j] = expf(S[i * n + j] - max_val);
            sum_exp += P[i * n + j];
        }
        
        for (int j = 0; j < n; j++) {
            P[i * n + j] /= sum_exp;
        }
    }

    // 3. 计算 O = P * V
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < d; k++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += P[i * n + j] * V[j * d + k];
            }
            O[i * d + k] = sum;
        }
    }

    free(S);
    free(P);
}

// FlashAttention 内核
__global__ void flash_attention_kernel(float *Q, float *K, float *V, float *O, float scale) {
    // 共享内存：用于存储 K 和 V 的分块
    // 大小为 [Bc][D]
    __shared__ float S_K[Bc][D];   
    __shared__ float S_V[Bc][D];

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    // 每个线程处理 Q 的一行
    int row_idx = bx * blockDim.x + tx;
    
    float my_q[D];
    float my_o[D];
    float l = 0.0f;
    float m = -CUDART_INF_F;

    // 1. 加载 Q 的一行到寄存器
    if (row_idx < N) {
        for (int i = 0; i < D; i++) {
            my_q[i] = Q[row_idx * D + i];
            my_o[i] = 0.0f; // 初始化输出
        }
    }

    // 2. 外层循环：遍历 K 和 V 的分块 (Tiling)
    // 每次处理 Bc 列
    for (int tile_idx = 0; tile_idx < (N + Bc - 1) / Bc; tile_idx++) {
        
        // --- 协同加载 K 和 V 的分块到共享内存 ---
        // 我们需要加载 Bc * D 个元素。
        // 线程块有 Br (128) 个线程。
        // 每个线程需要加载 (Bc * D) / Br 个元素。 why？
        // 32 * 64 / 128 = 16 个元素。
        
        int base_k_idx = tile_idx * Bc;
        
        for (int i = 0; i < (Bc * D) / Br; i++) {
            int total_idx = tx + i * Br; // 块内的线性索引
            int row = total_idx / D;
            int col = total_idx % D;
            
            if (row < Bc && (base_k_idx + row) < N) {
                S_K[row][col] = K[(base_k_idx + row) * D + col];
                S_V[row][col] = V[(base_k_idx + row) * D + col];
            } else {
                S_K[row][col] = 0.0f;
                S_V[row][col] = 0.0f;
            }
        }
        
        // 等待所有线程完成加载
        __syncthreads();

        // --- 计算 Attention ---
        if (row_idx < N) {
            // 遍历当前分块中的每一个 K 向量
            for (int j = 0; j < Bc; j++) {
                if (base_k_idx + j >= N) break;

                // a. 计算点积 score = Q[i] * K[j]
                float score = 0.0f;
                for (int k = 0; k < D; k++) {
                    score += my_q[k] * S_K[j][k];
                }
                score *= scale;

                // b. 在线 Softmax 更新 (Online Softmax Update)
                // 这是一个数值稳定的算法，可以在不知道全局最大值的情况下计算 Softmax
                
                float m_prev = m;
                float m_new = fmaxf(m, score);
                
                // 计算缩放因子
                float d_prev = expf(m_prev - m_new);
                float d_new = expf(score - m_new);
                
                // 更新分母 l
                l = l * d_prev + d_new;
                m = m_new;

                // 更新输出 O
                // O_new = (O_old * d_prev + V[j] * d_new)
                for (int k = 0; k < D; k++) {
                    my_o[k] = my_o[k] * d_prev + S_V[j][k] * d_new;
                }
            }
        }
        
        // 等待计算完成，以便下一轮加载可以覆盖共享内存
        __syncthreads();
    }

    // 3. 最终归一化并写入全局内存
    if (row_idx < N) {
        for (int i = 0; i < D; i++) {
            O[row_idx * D + i] = my_o[i] / l;
        }
    }
}

int main() {
    printf("=== CUDA FlashAttention (Simplified) ===\n\n");
    printf("配置: N=%d, D=%d, Br=%d, Bc=%d\n", N, D, Br, Bc);

    size_t size = N * D * sizeof(float);
    float *h_Q = (float*)malloc(size);
    float *h_K = (float*)malloc(size);
    float *h_V = (float*)malloc(size);
    float *h_O_gpu = (float*)malloc(size);
    float *h_O_cpu = (float*)malloc(size);

    srand(time(NULL));
    for (int i = 0; i < N * D; i++) {
        h_Q[i] = (float)(rand() % 100) / 100.0f;
        h_K[i] = (float)(rand() % 100) / 100.0f;
        h_V[i] = (float)(rand() % 100) / 100.0f;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, size);
    cudaMalloc(&d_K, size);
    cudaMalloc(&d_V, size);
    cudaMalloc(&d_O, size);

    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);

    // 缩放因子 1/sqrt(d)
    float scale = 1.0f / sqrtf((float)D);

    // 启动内核
    int num_blocks = (N + Br - 1) / Br;
    printf("启动内核: Grid=%d, Block=%d\n", num_blocks, Br);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    flash_attention_kernel<<<num_blocks, Br>>>(d_Q, d_K, d_V, d_O, scale);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU 执行时间: %.5f 毫秒\n", milliseconds);

    cudaMemcpy(h_O_gpu, d_O, size, cudaMemcpyDeviceToHost);

    // CPU 验证
    printf("正在进行 CPU 验证 (可能需要一点时间)...\n");
    attention_cpu(h_Q, h_K, h_V, h_O_cpu, N, D);

    // 检查误差
    float max_diff = 0.0f;
    for (int i = 0; i < N * D; i++) {
        float diff = fabs(h_O_gpu[i] - h_O_cpu[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("最大误差: %e\n", max_diff);
    if (max_diff < 1e-4) {
        printf("✓ 结果正确\n");
    } else {
        printf("✗ 结果可能有误\n");
    }

    free(h_Q); free(h_K); free(h_V); free(h_O_gpu); free(h_O_cpu);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}


✅ #include <stdlib.h> 的用途

提供 通用工具函数，包含：

👉 1. 内存分配

malloc()

calloc()

realloc()

free()

👉 2. 随机数

rand()

srand()

👉 3. 程序退出控制

exit()

atexit()

👉 4. 字符串转数字

atoi()

atof()

strtol() 等

👉 5. 常用系统工具函数

abs()

qsort()（快速排序）

✅ #include <stdio.h> 的用途

提供 输入输出函数：

👉 1. 文件与终端 I/O

printf()

scanf()

fprintf()

fscanf()

fopen() / fclose()

👉 2. 文件操作

fread()

fwrite()

fseek()

ftell()

👉 3. 字符 I/O

getchar()

putchar()`


##  float sum = 0.0f;  the sum is float32 
