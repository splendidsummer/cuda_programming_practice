#include <stdio.h>

#define N 4
/* 
float A[N][N] 这是一个二维数组的声明和定义。
float 表示数组元素类型是单精度浮点数（float）。
A[N][N] 表示 A 是一个 N 行 N 列的二维数组（矩阵），总共有 N×N 个 float 元素。
*/

__global__ void matrixAdd(float A[N][N], float B[N][N], float C[N][N]) {
    // 计算二维线程坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 列索引
    int y = blockIdx.y * blockDim.y + threadIdx.y; // 行索引

    if (x < N && y < N)
        C[y][x] = A[y][x] + B[y][x];
}

int main() {
    float A[N][N], B[N][N], C[N][N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }

    float (*d_A)[N], (*d_B)[N], (*d_C)[N];
    cudaMalloc(&d_A, N*N*sizeof(float));
    cudaMalloc(&d_B, N*N*sizeof(float));
    cudaMalloc(&d_C, N*N*sizeof(float));

    cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(2, 2);  // 每个block是 2×2 的线程
    dim3 numBlocks(N / 2, N / 2);
    matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%6.1f ", C[i][j]);
        printf("\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
