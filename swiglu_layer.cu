/*
 * CUDA 编程教程 - SwiGLU 神经网络层 (Full FFN Layer)
 *
 * 这是一个完整的 SwiGLU Feed-Forward Network (FFN) 层的实现，
 * 类似于 LLaMA 和其他现代 LLM 中使用的结构。
 *
 * 结构:
 * FFN(x) = (Swish(xW_g) * (xW_u)) W_d
 *
 * 包含三个步骤:
 * 1. 投影 (Projection): 使用 cuBLAS 进行矩阵乘法
 *    - Gate Proj: A = x * W_g
 *    - Up Proj:   B = x * W_u
 * 2. 激活 (Activation): 使用自定义 CUDA Kernel
 *    - C = Swish(A) * B
 * 3. 下投影 (Down Projection): 使用 cuBLAS
 *    - Out = C * W_d
 *
 * 编译 (需要链接 cuBLAS):
 * nvcc swiglu_layer.cu -o swiglu_layer -lcublas -O3
 * ./swiglu_layer
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define THREADS_PER_BLOCK 256

// ---------------------------------------------------------
// 1. SwiGLU 激活 Kernel (Element-wise)
// ---------------------------------------------------------
__global__ void swiglu_elementwise_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;
    
    if (vec_idx < n) {
        float4 g_val = reinterpret_cast<const float4*>(gate)[idx];
        float4 u_val = reinterpret_cast<const float4*>(up)[idx];
        float4 res;

        // x
        float val = g_val.x;
        float silu = val / (1.0f + expf(-val));
        res.x = silu * u_val.x;

        // y
        val = g_val.y;
        silu = val / (1.0f + expf(-val));
        res.y = silu * u_val.y;

        // z
        val = g_val.z;
        silu = val / (1.0f + expf(-val));
        res.z = silu * u_val.z;

        // w
        val = g_val.w;
        silu = val / (1.0f + expf(-val));
        res.w = silu * u_val.w;

        reinterpret_cast<float4*>(out)[idx] = res;
    }
}

// ---------------------------------------------------------
// 2. SwiGLU Layer 类定义
// ---------------------------------------------------------
class SwiGLULayer {
private:
    int batch_size;
    int input_dim;      // Hidden size (e.g., 4096)
    int hidden_dim;     // Intermediate size (e.g., 11008)
    
    // 权重 (Device)
    float *d_W_gate;    // [input_dim, hidden_dim]
    float *d_W_up;      // [input_dim, hidden_dim]
    float *d_W_down;    // [hidden_dim, input_dim]
    
    // 中间激活值 (Device)
    float *d_gate_proj; // [batch_size, hidden_dim]
    float *d_up_proj;   // [batch_size, hidden_dim]
    float *d_act_out;   // [batch_size, hidden_dim] (after swiglu)
    
    cublasHandle_t cublas_handle;

public:
    SwiGLULayer(int b, int in_d, int h_d) : batch_size(b), input_dim(in_d), hidden_dim(h_d) {
        cublasCreate(&cublas_handle);
        
        // 分配权重内存
        cudaMalloc(&d_W_gate, input_dim * hidden_dim * sizeof(float));
        cudaMalloc(&d_W_up,   input_dim * hidden_dim * sizeof(float));
        cudaMalloc(&d_W_down, hidden_dim * input_dim * sizeof(float));
        
        // 分配中间缓冲区
        cudaMalloc(&d_gate_proj, batch_size * hidden_dim * sizeof(float));
        cudaMalloc(&d_up_proj,   batch_size * hidden_dim * sizeof(float));
        cudaMalloc(&d_act_out,   batch_size * hidden_dim * sizeof(float));
        
        initialize_weights();
    }

    ~SwiGLULayer() {
        cudaFree(d_W_gate); cudaFree(d_W_up); cudaFree(d_W_down);
        cudaFree(d_gate_proj); cudaFree(d_up_proj); cudaFree(d_act_out);
        cublasDestroy(cublas_handle);
    }

    void initialize_weights() {
        // 简单随机初始化 (在实际应用中应从文件加载)
        int size_w1 = input_dim * hidden_dim;
        int size_w2 = hidden_dim * input_dim;
        
        float *h_temp = (float*)malloc(std::max(size_w1, size_w2) * sizeof(float));
        
        // Init W_gate
        for(int i=0; i<size_w1; i++) h_temp[i] = (float)(rand()%100)/1000.0f;
        cudaMemcpy(d_W_gate, h_temp, size_w1 * sizeof(float), cudaMemcpyHostToDevice);
        
        // Init W_up
        for(int i=0; i<size_w1; i++) h_temp[i] = (float)(rand()%100)/1000.0f;
        cudaMemcpy(d_W_up, h_temp, size_w1 * sizeof(float), cudaMemcpyHostToDevice);
        
        // Init W_down
        for(int i=0; i<size_w2; i++) h_temp[i] = (float)(rand()%100)/1000.0f;
        cudaMemcpy(d_W_down, h_temp, size_w2 * sizeof(float), cudaMemcpyHostToDevice);
        
        free(h_temp);
    }

    // 前向传播
    // input: [batch_size, input_dim]
    // output: [batch_size, input_dim]
    void forward(const float* d_input, float* d_output) {
        float alpha = 1.0f;
        float beta = 0.0f;

        // 1. Gate Projection: [batch, in] * [in, hidden] -> [batch, hidden]
        // cuBLAS 是列主序，所以计算 C = A * B 实际上是 C^T = B^T * A^T
        // 这里我们假设矩阵在内存中已经是合适的布局，或者我们使用转置标志
        // 简单起见，假设 input 是行主序 (batch, in)，权重是 (in, hidden)
        // 为了用 cuBLAS 计算 Row-Major 的 C = A * B:
        // C^T (Col-Major) = B^T * A^T
        // 实际上通常直接把 A 看作 (in, batch) 的列主序，或者使用 cublasSgemm 的转置能力
        
        // 标准做法: C = alpha * op(A) * op(B) + beta * C
        // A: input (batch, in), B: weight (in, hidden), C: out (batch, hidden)
        // cuBLAS 默认列主序。
        // 技巧: 计算 C^T = B^T * A^T (如果 A, B, C 都是行主序存储，这等价于 C = A * B)
        // 所以我们传: A=Weight^T (hidden, in), B=Input^T (in, batch), C=Out^T (hidden, batch)
        // 这里的 m=hidden_dim, n=batch_size, k=input_dim
        
        // Step 1.1: Gate Projection (X * W_g)
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    hidden_dim, batch_size, input_dim,
                    &alpha,
                    d_W_gate, hidden_dim,   // A (Weight is stored as hidden x in effectively if row major)
                    d_input, input_dim,     // B
                    &beta,
                    d_gate_proj, hidden_dim // C
        );

        // Step 1.2: Up Projection (X * W_u)
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    hidden_dim, batch_size, input_dim,
                    &alpha,
                    d_W_up, hidden_dim,
                    d_input, input_dim,
                    &beta,
                    d_up_proj, hidden_dim
        );

        // Step 2: SwiGLU Activation (Element-wise)
        int total_elements = batch_size * hidden_dim;
        int num_threads = total_elements / 4; // float4
        int num_blocks = (num_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        swiglu_elementwise_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
            d_gate_proj, d_up_proj, d_act_out, total_elements
        );

        // Step 3: Down Projection (Act * W_d)
        // Input: (batch, hidden), Weight: (hidden, in), Output: (batch, in)
        // m=input_dim, n=batch_size, k=hidden_dim
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    input_dim, batch_size, hidden_dim,
                    &alpha,
                    d_W_down, input_dim,
                    d_act_out, hidden_dim,
                    &beta,
                    d_output, input_dim
        );
    }
};

// ---------------------------------------------------------
// Main 测试
// ---------------------------------------------------------
int main() {
    printf("=== SwiGLU Full Layer Implementation ===\n");

    // 模拟一个小型的 Transformer 层参数
    int batch_size = 8;
    int seq_len = 128; // 这里我们将 batch * seq 视为总 batch
    int total_tokens = batch_size * seq_len;
    
    int input_dim = 1024;
    int hidden_dim = 4096; // 通常是 4 * input_dim 或 8/3 * input_dim

    printf("Tokens: %d, Input Dim: %d, Hidden Dim: %d\n", total_tokens, input_dim, hidden_dim);

    // 1. 准备输入数据
    size_t input_size = total_tokens * input_dim * sizeof(float);
    float *h_input = (float*)malloc(input_size);
    float *h_output = (float*)malloc(input_size); // Output has same shape as input

    for(int i=0; i<total_tokens * input_dim; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, input_size);
    
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);

    // 2. 创建层并运行
    SwiGLULayer layer(total_tokens, input_dim, hidden_dim);
    
    // Warmup
    layer.forward(d_input, d_output);
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for(int i=0; i<10; i++) {
        layer.forward(d_input, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("平均执行时间 (10次): %.3f ms\n", ms / 10.0f);

    // 3. 检查输出 (简单打印前几个值)
    cudaMemcpy(h_output, d_output, input_size, cudaMemcpyDeviceToHost);
    printf("Output[0..4]: %f, %f, %f, %f, %f\n", 
           h_output[0], h_output[1], h_output[2], h_output[3], h_output[4]);

    // Cleanup
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
