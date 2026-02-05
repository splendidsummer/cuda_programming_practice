#include <stdio.h> // 包含标准输入输出库，用于使用 printf

// CUDA kernel：向量加法，每个线程处理一个元素
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    // 计算全局线程索引（映射到数组下标）
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：仅当索引在数组范围内才执行运算
    if (i < N)
        C[i] = A[i] + B[i]; // 将对应元素相加并写回结果数组
}

int main() {
    int N = 16; // 向量长度（元素个数）
    float A[N], B[N], C[N]; // 在主机栈上分配三个数组：输入 A、B 和输出 C

    // 初始化输入数组 A 和 B
    for (int i = 0; i < N; i++) { A[i] = i; B[i] = 2*i; }

    // 设备端指针，用于在 GPU 上分配内存
    float *d_A, *d_B, *d_C;
    // 在设备上为 A、B、C 分配内存（字节大小为 N * sizeof(float)）
    cudaMalloc(&d_A, N*sizeof(float));
    cudaMalloc(&d_B, N*sizeof(float));
    cudaMalloc(&d_C, N*sizeof(float));

    // 将主机内存的数据拷贝到设备内存（Host -> Device）
    cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel：配置为每个 block 8 个线程，总共 N/8 个 blocks
    vectorAdd<<<N/8, 8>>>(d_A, d_B, d_C, N);

    // 将结果从设备拷贝回主机（Device -> Host）
    cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);

    // 输出结果数组 C 的每个元素
    for (int i = 0; i < N; i++)
        printf("C[%d] = %f\n", i, C[i]);

    // 释放设备端分配的内存
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}