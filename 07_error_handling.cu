/*
 * CUDA 编程教程 - 第7课：错误处理
 * 
 * 本示例展示如何正确处理 CUDA 错误：
 * 1. 检查 CUDA 函数返回值
 * 2. 使用错误检查宏
 * 3. 调试 CUDA 程序
 * 4. 常见错误类型
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA 错误在 %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// 检查内核启动错误
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "内核启动错误: %s\n", cudaGetErrorString(error)); \
            exit(1); \
        } \
        error = cudaDeviceSynchronize(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "内核执行错误: %s\n", cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// 简单的内核函数
__global__ void simple_kernel(float *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

// 可能导致错误的内核（访问越界）
__global__ void problematic_kernel(float *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 故意访问越界来演示错误
    data[idx] = data[idx] * 2.0f;  // 没有边界检查
}

// 演示内存不足错误
void demonstrate_out_of_memory()
{
    printf("--- 内存不足错误示例 ---\n");
    
    // 尝试分配非常大的内存
    size_t huge_size = 10ULL * 1024 * 1024 * 1024;  // 10 GB
    float *d_data;
    
    cudaError_t error = cudaMalloc((void**)&d_data, huge_size);
    if (error != cudaSuccess) {
        printf("预期的错误：%s\n", cudaGetErrorString(error));
        printf("错误代码：%d\n\n", error);
    } else {
        cudaFree(d_data);
    }
}

// 演示无效的内核配置
void demonstrate_invalid_config()
{
    printf("--- 无效内核配置示例 ---\n");
    
    float *d_data;
    CUDA_CHECK(cudaMalloc((void**)&d_data, 1024 * sizeof(float)));
    
    // 尝试使用无效的线程块大小
    // 每个线程块的最大线程数通常是 1024
    int invalid_threads = 2048;  // 超过限制
    
    printf("尝试启动内核，线程块大小：%d\n", invalid_threads);
    simple_kernel<<<1, invalid_threads>>>(d_data, 1024);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("检测到错误：%s\n", cudaGetErrorString(error));
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_data));
    printf("\n");
}

// 演示访问越界错误
void demonstrate_out_of_bounds()
{
    printf("--- 访问越界错误示例 ---\n");
    
    int n = 100;
    float *h_data = (float*)malloc(n * sizeof(float));
    float *d_data;
    
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i;
    }
    
    CUDA_CHECK(cudaMalloc((void**)&d_data, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice));
    
    // 启动过多的线程，导致访问越界
    int threads = 200;  // 超过数组大小
    printf("启动 %d 个线程处理 %d 个元素（可能越界）\n", threads, n);
    
    problematic_kernel<<<1, threads>>>(d_data, n);
    
    // 注意：访问越界可能不会立即报错，但会导致未定义行为
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);
    printf("（注意：访问越界可能不会报错，但会导致未定义行为）\n\n");
}

// 打印设备属性（检查设备能力）
void check_device_capabilities()
{
    printf("--- 检查设备能力 ---\n");
    
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        printf("错误：未检测到 CUDA 设备！\n");
        return;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("设备名称：%s\n", prop.name);
    printf("计算能力：%d.%d\n", prop.major, prop.minor);
    printf("每块最大线程数：%d\n", prop.maxThreadsPerBlock);
    printf("每块最大共享内存：%zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("全局内存：%.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("\n");
}

// 正确的错误处理示例
void correct_error_handling()
{
    printf("--- 正确的错误处理示例 ---\n");
    
    int n = 1024;
    float *h_data = (float*)malloc(n * sizeof(float));
    float *d_data;
    
    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i;
    }
    
    // 使用错误检查宏
    CUDA_CHECK(cudaMalloc((void**)&d_data, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice));
    
    // 配置内核
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // 启动内核
    simple_kernel<<<num_blocks, threads_per_block>>>(d_data, n);
    
    // 检查内核错误
    CUDA_CHECK_KERNEL();
    
    // 复制结果
    CUDA_CHECK(cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_data[i] - i * 2.0f) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("✓ 计算结果正确\n");
    } else {
        printf("✗ 计算结果有误\n");
    }
    
    // 清理
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);
    printf("\n");
}

int main()
{
    printf("=== CUDA 错误处理详解 ===\n\n");
    
    // 检查设备能力
    check_device_capabilities();
    
    // 正确的错误处理
    correct_error_handling();
    
    // 演示各种错误
    demonstrate_out_of_memory();
    demonstrate_invalid_config();
    demonstrate_out_of_bounds();
    
    printf("完成！\n");
    
    return 0;
}

/*
 * 编译和运行：
 * nvcc 07_error_handling.cu -o error_handling
 * ./error_handling
 * 
 * 错误处理最佳实践：
 * 1. 总是检查 CUDA 函数返回值
 * 2. 使用错误检查宏简化代码
 * 3. 在内核启动后检查 cudaGetLastError()
 * 4. 使用 cudaDeviceSynchronize() 确保内核完成
 * 5. 检查设备能力，确保内核配置有效
 * 
 * 常见错误类型：
 * 1. cudaErrorMemoryAllocation: 内存分配失败
 * 2. cudaErrorInvalidConfiguration: 无效的内核配置
 * 3. cudaErrorInvalidValue: 无效的参数值
 * 4. cudaErrorLaunchOutOfResources: 资源不足
 * 5. cudaErrorLaunchTimeout: 内核执行超时
 * 
 * 调试技巧：
 * 1. 使用 cuda-gdb 调试器
 * 2. 使用 Compute Sanitizer 检查内存错误
 * 3. 使用 nvcc -lineinfo 编译选项
 * 4. 添加打印语句（在内核中使用 printf）
 * 5. 逐步缩小问题范围
 * 
 * 编译调试版本：
 * nvcc -g -G -lineinfo 07_error_handling.cu -o error_handling
 * cuda-gdb ./error_handling
 */

