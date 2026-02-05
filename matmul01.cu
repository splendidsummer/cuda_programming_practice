#include <cublas_v2.h>    // 包含 cuBLAS 库的头文件（用于调用 cublasSgemm 等 API）
#include <cuda_runtime.h> // 包含 CUDA 运行时 API 的头文件

#include <cmath>    // for fabsf  // 包含数学函数（fabsf 用于比较浮点差异）
#include <fstream>  // for CSV output // 包含文件流，用于输出 CSV 文件
#include <iostream> // 包含标准输入输出流
#include <vector>   // 包含 std::vector 容器

#define TOL 1e-5f // 允许的误差阈值（比较结果时使用）

void checkCudaError(cudaError_t err, const char *msg) { // 检查 CUDA 错误并在出错时打印信息和退出
    if (err != cudaSuccess) {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl; // 输出错误信息
        exit(EXIT_FAILURE); // 退出程序
    }
}

void checkCublasError(cublasStatus_t status, const char *msg) { // 检查 cuBLAS 返回状态并在出错时打印信息和退出
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << " CUBLAS ERROR: " << status << std::endl; // 输出 cuBLAS 错误码
        exit(EXIT_FAILURE); // 退出程序
    }
}

template <const int BLOCK_SIZE>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, float *A, float *B,
                                                     float beta, float *C) { // 自定义的 tiled SGEMM kernel（模板参数为块大小）
    int bx = blockIdx.x; // 网格块在 x 方向的索引
    int by = blockIdx.y; // 网格块在 y 方向的索引

    const int BM = BLOCK_SIZE; // 每个 tile 在 M 方向的大小
    const int BN = BLOCK_SIZE; // 每个 tile 在 N 方向的大小
    const int BK = BLOCK_SIZE; // 每个 tile 在 K 方向的大小

    int tx = threadIdx.x % BN; // 线程在 tile 中的列索引（局部列）
    int ty = threadIdx.x / BN; // 线程在 tile 中的行索引（局部行）

    __shared__ float As[BM * BK]; // 用于存放 A 的 tile（共享内存）
    __shared__ float Bs[BK * BN]; // 用于存放 B 的 tile（共享内存）

    A = &A[by * BM * K]; // 将 A 指针移动到当前 tile 的起始行（按块定位 A）
    B = &B[bx * BN];     // 将 B 指针移动到当前 tile 的起始列（按块定位 B）
    C = &C[by * BM * N + bx * BN]; // 将 C 指针移动到当前 tile 的起始位置（按块定位 C）

    float tmp = 0.; // 局部累加变量，用于计算 C 的一个元素值
    for (int k = 0; k < K; k += BK) { // 沿 K 方向以 BK 为步长遍历 tiles
        As[ty * BK + tx] = A[ty * K + tx]; // 从全局内存加载 A 的子块到共享内存（按行主序偏移）
        Bs[ty * BN + tx] = B[ty * N + tx]; // 从全局内存加载 B 的子块到共享内存（按行主序偏移）
        __syncthreads(); // 同步线程，确保共享内存加载完成
        A += BK;        // A 指针向前移动 BK 列（进入下一段 K）
        B += BK * N;    // B 指针向前移动 BK 行（进入下一段 K）
        for (int i = 0; i < BK; i++) { // 在 tile 内进行乘加累加
            tmp += As[ty * BK + i] * Bs[i * BN + tx]; // 累加乘积
        }
        __syncthreads(); // 同步线程，确保在下一轮加载前所有线程都完成了读取共享内存
    }
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx]; // 写回结果到全局内存（带 alpha 和 beta）
}
#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N) // 向上取整除法宏，用于计算网格尺寸
int main() {
    std::vector<int> sizes = {1024}; // 要测试的矩阵大小列表（这里只测试 1024）

    // 打开CSV文件
    std::ofstream csv_file("sgemm_benchmark_v2.csv"); // 创建并打开输出 CSV 文件
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl; // 写入 CSV 表头

    for (int N : sizes) { // 遍历每个要测试的尺寸
        std::cout << "Testing size: " << N << std::endl; // 输出当前测试的尺寸

        size_t size = N * N * sizeof(float); // 矩阵占用的字节数
        float *A = (float *)malloc(size);    // 在主机上分配 A 矩阵
        float *B = (float *)malloc(size);    // 在主机上分配 B 矩阵
        float *C_cublas = (float *)malloc(size); // 在主机上分配用于保存 cuBLAS 结果的缓冲
        float *C_v1 = (float *)malloc(size);     // 在主机上分配用于保存自定义 kernel 结果的缓冲

        float *d_A, *d_B, *d_C_v1; // 设备指针
        checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed"); // 在设备上分配 d_A
        checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed"); // 在设备上分配 d_B
        checkCudaError(cudaMalloc(&d_C_v1, size), "cudaMalloc d_C_v1 failed"); // 在设备上分配 d_C_v1

        bool out_of_memory = false; // 标记是否发生 OOM 或其他错误

        try {
            // 初始化矩阵 A 和 B
            for (int i = 0; i < N * N; ++i) {
                A[i] = 1.0f; // 将 A 全置为 1.0
                B[i] = 2.0f; // 将 B 全置为 2.0
            }

            // 拷贝到设备
            checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice),
                                         "cudaMemcpy A to device failed"); // 将 A 复制到设备
            checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice),
                                         "cudaMemcpy B to device failed"); // 将 B 复制到设备

            cublasHandle_t handle; // cuBLAS 句柄
            checkCublasError(cublasCreate(&handle), "cublasCreate failed"); // 创建 cuBLAS 句柄

            float alpha = 1.0f; // alpha 标量
            float beta = 0.0f;  // beta 标量

            cudaEvent_t start, stop; // CUDA 事件用于计时
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed"); // 创建开始事件
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");   // 创建结束事件

            // warmup
            int warpup_time = 10;  // 热身次数（注意拼写为 warpup_time）
            for (int i = 0; i < warpup_time; ++i) {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                                                         &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                                                 "cublasSgemm failed"); // 使用 cuBLAS 执行 SGEMM 作为热身
            }
            cudaDeviceSynchronize(); // 等待设备完成热身调用

            // cuBLAS SGEMM
            int repeat_time = 5; // 重复次数以求平均
            checkCudaError(cudaEventRecord(start),
                                         "cudaEventRecord(start cublas) failed"); // 记录计时开始事件（cuBLAS）
            for (int i = 0; i < repeat_time; ++i) {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                                                         &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                                                 "cublasSgemm failed"); // 多次调用 cuBLAS SGEMM
            }

            checkCudaError(cudaEventRecord(stop),
                                         "cudaEventRecord(stop cublas) failed"); // 记录计时结束事件（cuBLAS）
            checkCudaError(cudaEventSynchronize(stop),
                                         "cudaEventSynchronize cublas failed"); // 等待结束事件完成

            float cublas_time = 0; // 存放 cuBLAS 总耗时（毫秒）
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                                         "cudaEventElapsedTime cublas failed"); // 计算时间差

            // 拷贝 cuBLAS 结果
            checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost),
                                         "cudaMemcpy C_cublas failed"); // 将 cuBLAS 的结果拷回主机

            // mysgemm_v1
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed"); // 将设备结果缓冲清零

            dim3 blockDim(1024); // 每个 block 的线程数（这里是 1024）
            dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(N, 32)); // 网格尺寸，每个维度按 32 的 tile 大小向上取整

            for (int i = 0; i < warpup_time; ++i) {
                mysgemm_v2<32>
                        <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1); // 调用自定义 kernel 作为热身
            }

            cudaDeviceSynchronize(); // 等待热身 kernel 完成
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed"); // 清零结果缓冲，准备计时测试

            checkCudaError(cudaEventRecord(start),
                                         "cudaEventRecord(start v1) failed"); // 记录自定义 kernel 计时开始事件

            for (int i = 0; i < repeat_time; ++i) {
                mysgemm_v2<32>
                        <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1); // 重复调用自定义 kernel
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed"); // 记录计时结束事件
            checkCudaError(cudaEventSynchronize(stop),
                                         "cudaEventSynchronize v1 failed"); // 等待结束事件完成
            float v1_time = 0; // 存放自定义 kernel 总耗时（毫秒）
            checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                                         "cudaEventElapsedTime v1 failed"); // 计算自定义 kernel 的耗时

            // 拷贝手写 kernel 结果
            checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                                         "cudaMemcpy C_v1 failed"); // 将自定义 kernel 的结果拷回主机
            // 结果比较
            int error_count = 0; // 记录不匹配的元素数（最多报告 10 个）
            for (int i = 0; i < N * N && error_count < 10; ++i) {
                if (fabsf(C_cublas[i] - C_v1[i]) > TOL) { // 比较元素差异是否超过阈值
                    error_count++; // 计数不匹配项
                }
            }

            float cublas_gflops =
                    repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);  // 计算 cuBLAS 的 GFLOPS（基于总耗时）
            float v1_gflops =
                    repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);  // 计算自定义 kernel 的 GFLOPS
            // 写入CSV
            csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
                             << (error_count == 0 ? "1" : "0") << std::endl; // 将本次结果写入 CSV（Matched 列为 1 或 0）

            // 释放资源
            cublasDestroy(handle); // 销毁 cuBLAS 句柄
            cudaEventDestroy(start); // 销毁事件 start
            cudaEventDestroy(stop);  // 销毁事件 stop
            cudaFree(d_A);           // 释放设备内存 d_A
            cudaFree(d_B);           // 释放设备内存 d_B
            cudaFree(d_C_v1);        // 释放设备内存 d_C_v1

            free(A);        // 释放主机内存 A
            free(B);        // 释放主机内存 B
            free(C_cublas); // 释放主机内存 C_cublas
            free(C_v1);     // 释放主机内存 C_v1

        } catch (...) { // 捕获所有异常（这里用于标记 OOM 等错误）
            std::cerr << "Out of memory or error during testing size: " << N
                                << std::endl; // 打印错误信息
            out_of_memory = true; // 标记为内存不足或其他错误
        }

        if (!out_of_memory) {
            std::cout << "Finished size: " << N << std::endl; // 如果成功完成，打印完成信息
        } else {
            csv_file << N << ",OOM,OOM,0" << std::endl; // 如果出错，在 CSV 中记录 OOM
        }
    }

    csv_file.close(); // 关闭 CSV 文件

    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'"
                        << std::endl; // 打印完成信息（注意文件名与实际写入的文件名略有不同）
    return 0; // 程序退出
}
