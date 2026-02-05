#include <cublas_v2.h> // 包含 cuBLAS 库的头文件，用于调用 cublasSgemm 等函数
#include <cuda_runtime.h> // 包含 CUDA 运行时 API 的头文件

#include <cmath>    // for fabsf // 包含数学函数 fabsf，用于比较浮点差异
#include <fstream>  // for CSV output // 包含文件流，用于输出 CSV 文件
#include <iostream> // 包含输入输出流，用于打印日志
#include <vector>   // 包含 vector 容器，用于保存测试规模列表

#define BLOCK_SIZE 32 // 定义线程块大小（每个维度的线程数），2维block的BLOCK_SIZE x BLOCK_SIZE  
#define TOL 1e-5f     // 定义比较结果的容差

void checkCudaError(cudaError_t err, const char *msg) { // 检查 CUDA API 返回的错误并打印
    if (err != cudaSuccess) { // 如果返回不是成功
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl; // 打印错误信息
        exit(EXIT_FAILURE); // 退出程序
    }
} // checkCudaError 结束

void checkCublasError(cublasStatus_t status, const char *msg) { // 检查 cuBLAS 返回状态并打印
    if (status != CUBLAS_STATUS_SUCCESS) { // 如果状态不是成功
        std::cerr << msg << " CUBLAS ERROR: " << status << std::endl; // 打印错误码（状态）
        exit(EXIT_FAILURE); // 退出程序
    }
} // checkCublasError 结束

// 手写的SGEMM kernel
__global__ void mysgemm_v1(int M, int N, int K, float alpha, float *A, float *B,
                                                     float beta, float *C) { // 自实现的矩阵乘法 kernel，计算 C = alpha * A * B + beta * C
    int gx = blockIdx.x * blockDim.x + threadIdx.x;  // 全局 x 坐标（列索引）
    int gy = blockIdx.y * blockDim.y + threadIdx.y;  // 全局 y 坐标（行索引）

    if (gx >= N || gy >= M) return; // 如果线程超出矩阵范围则返回

    float tmp = 0.0f; // 局部变量累加乘积
    for (int i = 0; i < K; i++) { // 遍历内维 K
        tmp += A[gy * K + i] * B[i * N + gx];  // 使用全局内存读取 A 和 B 的元素并累加
    }
    C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx]; // 写回结果到 C 矩阵（考虑 alpha 和 beta）
} // mysgemm_v1 kernel 结束

int main() { // 主函数入口
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192}; // 要测试的矩阵规模列表

    // 打开CSV文件
    std::ofstream csv_file("sgemm_benchmark_v1.csv"); // 创建并打开输出 CSV 文件
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl; // 写入 CSV 表头

    // // 传统写法
    // for (size_t idx = 0; idx < sizes.size(); ++idx) {
    // int N = sizes[idx];
    // 使用 N
    // }
    for (int N : sizes) { // 对每个规模进行循环测试
        std::cout << "Testing size: " << N << std::endl; // 打印当前测试规模

        size_t size = N * N * sizeof(float); // 单个矩阵所需字节数
        float *A = (float *)malloc(size); // 在主机上分配矩阵 A
        float *B = (float *)malloc(size); // 在主机上分配矩阵 B
        float *C_cublas = (float *)malloc(size); // 在主机上分配用于保存 cuBLAS 结果的矩阵
        float *C_v1 = (float *)malloc(size); // 在主机上分配用于保存自实现 kernel 结果的矩阵

        float *d_A, *d_B, *d_C_v1; // 设备指针声明
        checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed"); // 在设备上分配 A
        checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed"); // 在设备上分配 B
        checkCudaError(cudaMalloc(&d_C_v1, size), "cudaMalloc d_C_v1 failed"); // 在设备上分配 C 的存放空间

        bool out_of_memory = false; // 标志是否发生 OOM 或其他错误

        try { // 使用 try/catch 包裹资源使用过程（代码依赖于异常捕获）
            // 初始化矩阵 A 和 B
            for (int i = 0; i < N * N; ++i) { // 遍历所有元素
                A[i] = 1.0f; // 将 A 初始化为 1.0
                B[i] = 2.0f; // 将 B 初始化为 2.0
            }

            // 拷贝到设备
            checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice),
                                         "cudaMemcpy A to device failed"); // 将 A 拷贝到设备
            checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice),
                                         "cudaMemcpy B to device failed"); // 将 B 拷贝到设备
            /*
                cublasHandle_t 是一个不透明的数据结构（您不需要知道它内部具体有什么），
                它存储了 cuBLAS 库运行所需的所有状态信息和上下文。
                当您创建一个句柄时，cuBLAS 会在后台为您设置好一个完整的工作环境。这个环境包括：
                关联的 CUDA 上下文：指明该句柄将在哪个 GPU 上工作。
                流（Stream）管理：记录计算任务将在哪个 CUDA 流中执行（这关系到任务并行）。
                数学模式设置：例如是否使用 Tensor Cores，是否允许快速但精度稍低的数学运算等。
                指针模式设置：控制标量参数（如 alpha, beta）是来自主机内存还是设备内存。
                内部库资源：如为了高效运行而预先分配的临时工作区、内核函数配置等。
            */

            cublasHandle_t handle; // cuBLAS 句柄声明
            checkCublasError(cublasCreate(&handle), "cublasCreate failed"); // 创建 cuBLAS 句柄

            float alpha = 1.0f; // alpha 参数
            float beta = 0.0f; // beta 参数

            cudaEvent_t start, stop; // CUDA 事件用于计时
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed"); // 创建开始事件
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed"); // 创建结束事件

            // warmup
            int warpup_time = 10;  // 热身次数
            for (int i = 0; i < warpup_time; ++i) { // 执行若干次 cuBLAS 以热身 GPU
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                                                         &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                                                 "cublasSgemm failed"); // 调用 cuBLAS 执行矩阵乘法（注意参数顺序）
            }
            cudaDeviceSynchronize(); // 同步设备，确保热身完成

            // cuBLAS SGEMM
            int repeat_time = 5; // 计时时重复次数
            checkCudaError(cudaEventRecord(start),
                                         "cudaEventRecord(start cublas) failed"); // 记录开始事件
            for (int i = 0; i < repeat_time; ++i) { // 重复调用 cuBLAS，用于计时平均
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                                                         &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                                                 "cublasSgemm failed"); // 调用 cuBLAS SGEMM
            }

            checkCudaError(cudaEventRecord(stop),
                                         "cudaEventRecord(stop cublas) failed"); // 记录结束事件
            checkCudaError(cudaEventSynchronize(stop),
                                         "cudaEventSynchronize cublas failed"); // 等待事件完成

            float cublas_time = 0; // 存放 cuBLAS 花费时间（毫秒）
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                                         "cudaEventElapsedTime cublas failed"); // 计算时间差

            // 拷贝 cuBLAS 结果
            checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost),
                                         "cudaMemcpy C_cublas failed"); // 将 cuBLAS 输出拷回主机

            // mysgemm_v1
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed"); // 将设备上的 C 初始化为 0

            dim3 threads(BLOCK_SIZE, BLOCK_SIZE); // 定义线程块维度
            dim3 blocks((N + threads.x - 1) / threads.x,
                                    (N + threads.y - 1) / threads.y); // 计算需要的线程块数量以覆盖 N x N

            for (int i = 0; i < warpup_time; ++i) { // 热身自实现 kernel
                mysgemm_v1<<<blocks, threads>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1); // 启动 kernel
            }
            cudaDeviceSynchronize(); // 同步 GPU，确保热身完成

            checkCudaError(cudaEventRecord(start),
                                         "cudaEventRecord(start v1) failed"); // 记录自实现计时开始事件
            for (int i = 0; i < repeat_time; ++i) { // 重复调用自实现 kernel
                mysgemm_v1<<<blocks, threads>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1); // 启动 kernel
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed"); // 记录自实现计时结束事件
            checkCudaError(cudaEventSynchronize(stop),
                                         "cudaEventSynchronize v1 failed"); // 同步事件

            float v1_time = 0; // 存放自实现 kernel 的耗时（毫秒）
            checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                                         "cudaEventElapsedTime v1 failed"); // 计算时间差

            // 拷贝手写 kernel 结果
            checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                                         "cudaMemcpy C_v1 failed"); // 将自实现结果拷回主机
            // 结果比较
            int error_count = 0; // 记录不匹配的元素数量
            for (int i = 0; i < N * N && error_count < 10; ++i) { // 遍历元素，最多记录前 10 个错误
                if (fabsf(C_cublas[i] - C_v1[i]) > TOL) { // 如果差值超过容差
                    error_count++; // 增加错误计数
                }
            }

            float cublas_gflops =
                    repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);  // GFlops 计算（cuBLAS）
            float v1_gflops =
                    repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);  // GFlops 计算（自实现）
            // 写入CSV
            csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
                             << (error_count == 0 ? "1" : "0") << std::endl; // 将结果写入 CSV（Matched 为 1 或 0）

            // 释放资源
            cublasDestroy(handle); // 销毁 cuBLAS 句柄
            cudaEventDestroy(start); // 销毁开始事件
            cudaEventDestroy(stop); // 销毁结束事件
            cudaFree(d_A); // 释放设备内存 A
            cudaFree(d_B); // 释放设备内存 B
            cudaFree(d_C_v1); // 释放设备内存 C

            free(A); // 释放主机内存 A
            free(B); // 释放主机内存 B
            free(C_cublas); // 释放主机内存 C_cublas
            free(C_v1); // 释放主机内存 C_v1

        } catch (...) { // 捕获所有异常
            std::cerr << "Out of memory or error during testing size: " << N
                                << std::endl; // 打印错误信息
            out_of_memory = true; // 设置 OOM 标志
        }

        if (!out_of_memory) { // 如果没有 OOM
            std::cout << "Finished size: " << N << std::endl; // 打印完成信息
        } else {
            csv_file << N << ",OOM,OOM,0" << std::endl; // 写入 CSV 标记 OOM
        }
    } // sizes 循环结束

    csv_file.close(); // 关闭 CSV 文件

    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'"
                        << std::endl; // 打印完成消息（注意文件名与实际写入可能不同）
    return 0; // 程序正常退出
} // main 结束
