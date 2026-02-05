/*
 * CUDA 编程示例：稀疏矩阵向量乘法 (SpMV) - CSR 格式
 * 
 * 这个示例展示了在真实场景中 "indices" (索引数组) 是如何产生的以及如何使用的。
 * 
 * 场景：
 * 我们要计算 y = A * x
 * 其中 A 是一个稀疏矩阵（大部分元素为0），x 是一个稠密向量。
 * 
 * 数据结构 (CSR - Compressed Sparse Row):
 * 为了节省内存，我们不存储 0 元素。我们使用三个数组来表示矩阵 A：
 * 1. values[]:      存储所有非零元素的值。
 * 2. col_indices[]: 存储每个非零元素对应的列号 (这就是本例中的 "indices")。
 * 3. row_ptr[]:     存储每一行在 values 数组中的起始位置。
 * 
 * Gather 操作发生在哪里？
 * 在计算点积时，我们需要用矩阵元素 A[i][j] 乘以向量元素 x[j]。
 * 由于 A 是压缩存储的，我们不知道 j 是多少，必须从 col_indices 数组中读取。
 * 然后根据读取到的列号 j，去向量 x 中 "收集" (Gather) 对应的值 x[j]。
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 简单的 CSR SpMV 核函数
// 每个线程负责计算结果向量 y 中的一行
__global__ void spmv_csr_kernel(const float *values, 
                                const int *col_indices, 
                                const int *row_ptr, 
                                const float *x, 
                                float *y, 
                                int num_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        float dot = 0.0f;
        
        // 获取当前行在 values 和 col_indices 数组中的起始和结束位置
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        // 遍历当前行的所有非零元素
        for (int i = row_start; i < row_end; i++) {
            // 1. 读取列索引 (这就是 "indices" 的作用)
            // 这是一个预先计算好的索引，告诉我们这个非零元素在原矩阵的第几列
            int col = col_indices[i]; 
            
            // 2. 读取非零元素的值
            float val = values[i];
            
            // 3. Gather (收集) 操作
            // 根据列索引 col，从向量 x 中读取对应的值
            // 这就是典型的间接寻址： x[col_indices[i]]
            dot += val * x[col];
        }

        // 写入结果
        y[row] = dot;
    }
}

// 辅助函数：打印数组
void print_array(const char *name, float *arr, int n) {
    printf("%s: [ ", name);
    for (int i = 0; i < n; i++) {
        printf("%.1f ", arr[i]);
    }
    printf("]\n");
}

int main() {
    printf("=== CUDA 稀疏矩阵向量乘法 (CSR SpMV) 示例 ===\n");
    printf("展示 indices 数组在真实计算中的用途\n\n");

    // ---------------------------------------------------------
    // 1. 定义一个 4x4 的稀疏矩阵 A 和向量 x
    // ---------------------------------------------------------
    /*
       矩阵 A (4行4列):
       [ 10,  0, 20,  0 ]  <- 第0行: 非零元素在列 0, 2
       [  0, 30,  0, 40 ]  <- 第1行: 非零元素在列 1, 3
       [ 50, 60, 70, 80 ]  <- 第2行: 非零元素在列 0, 1, 2, 3 (稠密行)
       [  0,  0,  0, 90 ]  <- 第3行: 非零元素在列 3
    */
    int num_rows = 4;
    int num_cols = 4;
    int num_non_zeros = 9;

    // 向量 x
    float h_x[] = {1.0f, 2.0f, 3.0f, 4.0f};

    // ---------------------------------------------------------
    // 2. 将矩阵 A 转换为 CSR 格式 (预处理阶段)
    // ---------------------------------------------------------
    // 这些数据通常是从文件读取或由其他程序生成的，而不是随机生成的
    
    // values: 所有非零元素的值，按行顺序排列
    float h_values[] = {
        10.0f, 20.0f,        // 第0行
        30.0f, 40.0f,        // 第1行
        50.0f, 60.0f, 70.0f, 80.0f, // 第2行
        90.0f                // 第3行
    };

    // col_indices: 对应 values 中每个元素的列号
    // 这就是我们要演示的 "indices" 数组
    int h_col_indices[] = {
        0, 2,          // 10在第0列, 20在第2列
        1, 3,          // 30在第1列, 40在第3列
        0, 1, 2, 3,    // 50在第0列...
        3              // 90在第3列
    };

    // row_ptr: 每一行在 values/col_indices 数组中的起始索引
    // 最后一个元素是总非零元素个数
    int h_row_ptr[] = {0, 2, 4, 8, 9};

    // 结果向量 y
    float h_y[4] = {0};

    printf("矩阵 A (CSR格式):\n");
    printf("Values:      "); for(int i=0; i<num_non_zeros; i++) printf("%.0f ", h_values[i]); printf("\n");
    printf("Col Indices: "); for(int i=0; i<num_non_zeros; i++) printf("%d ", h_col_indices[i]); printf("\n");
    printf("Row Ptr:     "); for(int i=0; i<=num_rows; i++) printf("%d ", h_row_ptr[i]); printf("\n\n");
    
    print_array("向量 x", h_x, num_cols);
    printf("\n");

    // ---------------------------------------------------------
    // 3. 分配 GPU 内存
    // ---------------------------------------------------------
    float *d_values, *d_x, *d_y;
    int *d_col_indices, *d_row_ptr;

    cudaMalloc((void**)&d_values, num_non_zeros * sizeof(float));
    cudaMalloc((void**)&d_col_indices, num_non_zeros * sizeof(int));
    cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_x, num_cols * sizeof(float));
    cudaMalloc((void**)&d_y, num_rows * sizeof(float));

    // ---------------------------------------------------------
    // 4. 复制数据到 GPU
    // ---------------------------------------------------------
    cudaMemcpy(d_values, h_values, num_non_zeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, h_col_indices, num_non_zeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, h_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, num_cols * sizeof(float), cudaMemcpyHostToDevice);

    // ---------------------------------------------------------
    // 5. 执行 Kernel
    // ---------------------------------------------------------
    // 只需要一个 Block，包含 num_rows 个线程 (这里是4个)
    spmv_csr_kernel<<<1, num_rows>>>(d_values, d_col_indices, d_row_ptr, d_x, d_y, num_rows);
    cudaDeviceSynchronize();

    // ---------------------------------------------------------
    // 6. 获取结果并验证
    // ---------------------------------------------------------
    cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    printf("计算结果 y = A * x:\n");
    // 手动计算验证:
    // y[0] = 10*x[0] + 20*x[2] = 10*1 + 20*3 = 70
    // y[1] = 30*x[1] + 40*x[3] = 30*2 + 40*4 = 60 + 160 = 220
    // y[2] = 50*1 + 60*2 + 70*3 + 80*4 = 50 + 120 + 210 + 320 = 700
    // y[3] = 90*x[3] = 90*4 = 360
    
    for (int i = 0; i < num_rows; i++) {
        printf("Row %d: %.1f\n", i, h_y[i]);
    }

    // 清理内存
    cudaFree(d_values);
    cudaFree(d_col_indices);
    cudaFree(d_row_ptr);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
