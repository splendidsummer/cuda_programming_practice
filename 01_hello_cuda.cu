/*
 * CUDA 编程教程 - 第1课：Hello CUDA
 * 
 * 这是你的第一个 CUDA 程序。它将展示如何：
 * 1. 创建一个简单的 CUDA 内核函数
 * 2. 从主机（CPU）调用设备（GPU）上的内核
 * 3. 配置线程块和网格
 */

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA 内核函数：在 GPU 上执行的函数
// __global__ 修饰符表示这是一个可以从 CPU 调用的 GPU 函数
__global__ void hello_cuda()
{
    // 获取当前线程的索引
    int thread_id = threadIdx.x;  // 线程在线程块中的索引
    int block_id = blockIdx.x;    // 线程块在网格中的索引
    
    // 计算全局线程 ID
    int global_id = block_id * blockDim.x + thread_id;
    
    printf("Hello from GPU! Thread %d in Block %d (Global ID: %d)\n", 
           thread_id, block_id, global_id);
}

int main()
{
    printf("=== CUDA Hello World 示例 ===\n\n");
    
    // 配置内核启动参数
    // 格式：<<<网格大小, 线程块大小>>>
    // 这里创建 2 个线程块，每个线程块包含 4 个线程
    dim3 num_blocks(10, 10);    // 网格中的线程块数量
    int threads_per_block = 4;  // 每个线程块中的线程数量
    
    printf("启动内核：%d 个线程块，每个块 %d 个线程\n", 
           num_blocks.x * num_blocks.y, threads_per_block);
    printf("总线程数：%d\n\n", num_blocks.x * num_blocks.y * threads_per_block);
    
    // 调用 CUDA 内核
    // <<<网格配置, 线程块配置>>>
    hello_cuda<<<num_blocks, threads_per_block>>>();
    
    // 等待 GPU 完成所有操作
    // 这是必要的，因为内核调用是异步的
    cudaDeviceSynchronize();
    
    // 检查是否有错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA 错误: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    printf("\n内核执行完成！\n");
    
    return 0;
}

/*
 * 编译和运行：
 * nvcc 01_hello_cuda.cu -o hello_cuda
 * ./hello_cuda
 * 
 * 关键概念：
 * 1. __global__: 内核函数修饰符，可以从 CPU 调用，在 GPU 上执行
 * 2. threadIdx.x: 线程在线程块中的 x 维度索引
 * 3. blockIdx.x: 线程块在网格中的 x 维度索引
 * 4. blockDim.x: 线程块在 x 维度的大小
 * 5. <<< >>>: 内核启动配置语法
 * 6. cudaDeviceSynchronize(): 同步主机和设备，等待 GPU 完成
 */

