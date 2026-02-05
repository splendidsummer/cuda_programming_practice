/*
 * CUDA 编程教程 - 第6课：内存管理
 * 
 * 本示例展示 CUDA 中的各种内存类型：
 * 1. 全局内存 (Global Memory)
 * 2. 共享内存 (Shared Memory)
 * 3. 常量内存 (Constant Memory)
 * 4. 纹理内存 (Texture Memory)
 * 5. 寄存器 (Registers)
 * 6. 统一内存 (Unified Memory)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024
#define CONSTANT_SIZE 64

// 常量内存：在编译时确定，只读，有缓存
__constant__ float constant_array[CONSTANT_SIZE];

// 全局内存：所有线程都可以访问
__global__ void use_global_memory(float *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // 访问全局内存
    }
}

// 共享内存：线程块内共享
__global__ void use_shared_memory(float *input, float *output, int n)
{
    __shared__ float shared_data[256];  // 共享内存
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 从全局内存加载到共享内存
    if (idx < n) {
        shared_data[tid] = input[idx];
    } else {
        shared_data[tid] = 0.0f;
    }
    
    __syncthreads();  // 同步
    
    // 处理共享内存中的数据
    shared_data[tid] = shared_data[tid] * 3.0f;
    
    __syncthreads();  // 同步
    
    // 写回全局内存
    if (idx < n) {
        output[idx] = shared_data[tid];
    }
}

// 常量内存：只读，有缓存优化
__global__ void use_constant_memory(float *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 常量内存对所有线程都是只读的，有缓存
        output[idx] = constant_array[idx % CONSTANT_SIZE] + idx;
    }
}

// 统一内存：自动管理主机和设备内存
__global__ void use_unified_memory(float *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 5.0f;
    }
}

// 获取设备信息
void print_device_info()
{
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    printf("=== 设备信息 ===\n");
    printf("CUDA 设备数量：%d\n\n", device_count);
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("设备 %d: %s\n", i, prop.name);
        printf("  计算能力：%d.%d\n", prop.major, prop.minor);
        printf("  全局内存：%.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  共享内存每块：%zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  每块最大线程数：%d\n", prop.maxThreadsPerBlock);
        printf("  每 SM 最大线程数：%d\n", prop.maxThreadsPerMultiProcessor);
        printf("  SM 数量：%d\n", prop.multiProcessorCount);
        printf("  常量内存：%zu KB\n", prop.totalConstMem / 1024);
        printf("  寄存器每块：%d\n", prop.regsPerBlock);
        printf("\n");
    }
}

int main()
{
    printf("=== CUDA 内存管理详解 ===\n\n");
    
    // 打印设备信息
    print_device_info();
    
    // ========== 全局内存示例 ==========
    printf("--- 全局内存示例 ---\n");
    float *h_data = (float*)malloc(N * sizeof(float));
    float *d_data;
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    
    // 分配全局内存
    cudaMalloc((void**)&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动内核
    use_global_memory<<<(N + 255) / 256, 256>>>(d_data, N);
    cudaDeviceSynchronize();
    
    // 复制结果回主机
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("结果（前 10 个元素）：");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_data[i]);
    }
    printf("\n\n");
    
    // ========== 共享内存示例 ==========
    printf("--- 共享内存示例 ---\n");
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));
    
    // 重新初始化
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    cudaMemcpy(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 使用共享内存
    use_shared_memory<<<(N + 255) / 256, 256>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_data, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("结果（前 10 个元素）：");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_data[i]);
    }
    printf("\n\n");
    
    // ========== 常量内存示例 ==========
    printf("--- 常量内存示例 ---\n");
    float h_constant[CONSTANT_SIZE];
    for (int i = 0; i < CONSTANT_SIZE; i++) {
        h_constant[i] = (float)(i * 2);
    }
    
    // 复制到常量内存
    cudaMemcpyToSymbol(constant_array, h_constant, CONSTANT_SIZE * sizeof(float));
    
    // 使用常量内存
    use_constant_memory<<<(N + 255) / 256, 256>>>(d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_data, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("结果（前 10 个元素）：");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_data[i]);
    }
    printf("\n\n");
    
    // ========== 统一内存示例 ==========
    printf("--- 统一内存示例 ---\n");
    float *unified_data;
    
    // 分配统一内存（CUDA 6.0+）
    cudaMallocManaged(&unified_data, N * sizeof(float));
    
    // 初始化数据（在主机上）
    for (int i = 0; i < N; i++) {
        unified_data[i] = (float)i;
    }
    
    // 启动内核（不需要显式复制）
    use_unified_memory<<<(N + 255) / 256, 256>>>(unified_data, N);
    cudaDeviceSynchronize();
    
    // 直接访问结果（不需要显式复制）
    printf("结果（前 10 个元素）：");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", unified_data[i]);
    }
    printf("\n\n");
    
    // ========== 内存使用统计 ==========
    printf("--- 内存使用统计 ---\n");
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    printf("GPU 总内存：%.2f MB\n", total_bytes / (1024.0 * 1024.0));
    printf("GPU 可用内存：%.2f MB\n", free_bytes / (1024.0 * 1024.0));
    printf("GPU 已用内存：%.2f MB\n", 
           (total_bytes - free_bytes) / (1024.0 * 1024.0));
    
    // 清理
    cudaFree(d_data);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(unified_data);
    free(h_data);
    
    printf("\n完成！\n");
    
    return 0;
}

/*
 * 编译和运行：
 * nvcc 06_memory_management.cu -o memory_management
 * ./memory_management
 * 
 * 内存类型总结：
 * 1. 全局内存 (Global Memory)
 *    - 容量大，所有线程可访问
 *    - 延迟高，带宽有限
 *    - 使用 cudaMalloc 分配
 * 
 * 2. 共享内存 (Shared Memory)
 *    - 线程块内共享
 *    - 速度快，容量小（通常 48KB）
 *    - 使用 __shared__ 声明
 * 
 * 3. 常量内存 (Constant Memory)
 *    - 只读，有缓存
 *    - 容量小（64KB）
 *    - 使用 __constant__ 声明
 *    - 使用 cudaMemcpyToSymbol 复制数据
 * 
 * 4. 纹理内存 (Texture Memory)
 *    - 只读，有缓存
 *    - 适合二维空间局部性访问
 *    - （本示例未包含，较高级）
 * 
 * 5. 寄存器 (Registers)
 *    - 最快，线程私有
 *    - 数量有限
 *    - 自动管理
 * 
 * 6. 统一内存 (Unified Memory)
 *    - CUDA 6.0+ 特性
 *    - 自动管理主机和设备内存
 *    - 使用 cudaMallocManaged 分配
 *    - 简化内存管理，但可能有性能开销
 * 
 * 内存访问优化：
 * - 合并访问（Coalesced Access）：连续内存访问
 * - 使用共享内存减少全局内存访问
 * - 合理使用常量内存缓存
 * - 避免内存碎片
 */

