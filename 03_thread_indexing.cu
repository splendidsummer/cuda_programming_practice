/*
 * CUDA 编程教程 - 第3课：线程索引详解
 * 
 * 本示例深入讲解 CUDA 线程索引系统：
 * 1. threadIdx: 线程在线程块中的索引
 * 2. blockIdx: 线程块在网格中的索引
 * 3. blockDim: 线程块的维度
 * 4. gridDim: 网格的维度
 * 5. 一维、二维、三维索引
 */

#include <stdio.h>
#include <cuda_runtime.h>

/*
 * CUDA 编程教程 - 第3课：线程索引详解
 * 
 * 本示例深入讲解 CUDA 线程索引系统：
 * 1. threadIdx: 线程在线程块中的索引
 * 2. blockIdx: 线程块在网格中的索引
 * 3. blockDim: 线程块的维度
 * 4. gridDim: 网格的维度
 * 5. 一维、二维、三维索引
 */

#include <stdio.h>                         // 引入标准输入输出库
#include <cuda_runtime.h>                  // 引入 CUDA 运行时头文件

// 一维线程索引示例
__global__ void print_1d_index()           // 定义打印一维索引的核函数
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                            // 计算全局线程索引
    printf("1D: Block %d, Thread %d -> Global ID %d\n", 
           blockIdx.x, threadIdx.x, tid);                                       // 打印块索引、线程索引和全局索引
}                                                                               // print_1d_index 结束

// 二维线程索引示例
__global__ void print_2d_index()                                               // 定义打印二维索引的核函数
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;                            // 计算全局行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;                            // 计算全局列索引
    
    int tid = row * gridDim.x * blockDim.x + col;                               // 将二维索引转换成线性索引
    
    printf("2D: Block(%d,%d), Thread(%d,%d) -> Global(%d,%d) -> Linear %d\n",
           blockIdx.x, blockIdx.y, 
           threadIdx.x, threadIdx.y,
           col, row, tid);                                                      // 打印二维索引和线性索引
}                                                                               // print_2d_index 结束

// 矩阵元素处理示例
__global__ void matrix_process(float *matrix, int width, int height)            // 定义矩阵处理核函数
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;                            // 计算当前线程负责的行
    int col = blockIdx.x * blockDim.x + threadIdx.x;                            // 计算当前线程负责的列
    
    if (row < height && col < width) {                                          // 边界检查
        int index = row * width + col;                                          // 计算线性索引
        matrix[index] = row * 100.0f + col;                                     // 为矩阵元素赋值
    }                                                                           // if 结束
}                                                                               // matrix_process 结束

int main()                                                                      // 程序入口
{
    printf("=== CUDA 线程索引详解 ===\n\n");                                    // 打印标题
    
    printf("--- 一维索引示例 ---\n");                                           // 打印一维示例标题
    printf("配置：2 个线程块，每个块 3 个线程\n\n");                           // 打印配置说明
    print_1d_index<<<2, 3>>>();                                                 // 启动一维打印核函数
    cudaDeviceSynchronize();                                                    // 等待设备完成
    printf("\n");                                                               // 输出空行
    
    printf("--- 二维索引示例 ---\n");                                           // 打印二维示例标题
    printf("配置：2x2 网格，每个块 2x2 线程\n\n");                             // 打印配置说明
    
    dim3 grid_size(2, 2);                                                       // 定义 2x2 网格
    dim3 block_size(2, 2);                                                      // 定义 2x2 线程块
    
    print_2d_index<<<grid_size, block_size>>>();                                // 启动二维打印核函数
    cudaDeviceSynchronize();                                                    // 等待设备完成
    printf("\n");                                                               // 输出空行
    
    printf("--- 矩阵处理示例 ---\n");                                           // 打印矩阵示例标题
    int width = 5;                                                              // 定义矩阵宽度
    int height = 4;                                                             // 定义矩阵高度
    int size = width * height;                                                  // 计算总元素数量
    
    float *h_matrix = (float*)malloc(size * sizeof(float));                     // 在主机端分配矩阵内存
    
    float *d_matrix;                                                            // 设备端矩阵指针
    cudaMalloc((void**)&d_matrix, size * sizeof(float));                        // 在设备端分配矩阵内存
    
    int block_width = 16;                                                       // 线程块宽度
    int block_height = 16;                                                      // 线程块高度
    
    dim3 grid((width + block_width - 1) / block_width,
              (height + block_height - 1) / block_height);                      // 计算网格尺寸
    dim3 block(block_width, block_height);                                      // 配置线程块尺寸
    
    printf("矩阵大小：%d x %d\n", width, height);                               // 打印矩阵大小
    printf("网格大小：%d x %d 线程块\n", grid.x, grid.y);                       // 打印网格信息
    printf("线程块大小：%d x %d 线程\n", block.x, block.y);                     // 打印线程块信息
    
    matrix_process<<<grid, block>>>(d_matrix, width, height);                   // 启动矩阵处理核函数
    cudaDeviceSynchronize();                                                    // 等待核函数完成
    
    cudaMemcpy(h_matrix, d_matrix, size * sizeof(float), cudaMemcpyDeviceToHost); // 将结果拷回主机
    
    printf("\n结果矩阵：\n");                                                    // 打印结果标题
    for (int i = 0; i < height; i++) {                                          // 遍历行
        for (int j = 0; j < width; j++) {                                       // 遍历列
            printf("%6.1f ", h_matrix[i * width + j]);                          // 打印元素
        }                                                                       // 内层循环结束
        printf("\n");                                                           // 换行
    }                                                                           // 外层循环结束
    
    cudaFree(d_matrix);                                                         // 释放设备内存
    free(h_matrix);                                                             // 释放主机内存
    
    printf("\n完成！\n");                                                        // 打印完成信息
    
    return 0;                                                                   // 返回 0 表示成功
}                                                                               // main 结束

/*
 * 编译和运行：
 * nvcc 03_thread_indexing.cu -o thread_indexing
 * ./thread_indexing
 * 
 * 关键概念：
 * 1. threadIdx: 三维向量 (x, y, z)，线程在线程块中的位置
 * 2. blockIdx: 三维向量 (x, y, z)，线程块在网格中的位置
 * 3. blockDim: 三维向量，线程块的维度（线程数）
 * 4. gridDim: 三维向量，网格的维度（线程块数）
 * 
 * 一维全局索引：
 *   tid = blockIdx.x * blockDim.x + threadIdx.x
 * 
 * 二维全局索引：
 *   row = blockIdx.y * blockDim.y + threadIdx.y
 *   col = blockIdx.x * blockDim.x + threadIdx.x
 * 
 * 线性索引（行主序）：
 *   index = row * width + col
 */

// ========== 3D 索引与 3D 张量示例 ==========

// 打印 3D 线程/块索引并给出线性全局索引
__global__ void print_3d_index()                                               // 定义 3D 索引打印核函数
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;                             // 计算全局 x
    int gy = blockIdx.y * blockDim.y + threadIdx.y;                             // 计算全局 y
    int gz = blockIdx.z * blockDim.z + threadIdx.z;                             // 计算全局 z

    int total_x = gridDim.x * blockDim.x;                                       // 整个网格的 x 方向线程数
    int total_y = gridDim.y * blockDim.y;                                       // 整个网格的 y 方向线程数
    int linear = gz * (total_y * total_x) + gy * total_x + gx;                  // 计算线性索引

    printf("3D: Block(%d,%d,%d) Thread(%d,%d,%d) -> Global(%d,%d,%d) -> Linear %d\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           gx, gy, gz, linear);                                                 // 打印 3D 信息
}                                                                               // print_3d_index 结束

__global__ void tensor3d_process(float *tensor, int width, int height, int depth) // 定义 3D 张量处理核函数
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;                              // 计算全局 x
    int y = blockIdx.y * blockDim.y + threadIdx.y;                              // 计算全局 y
    int z = blockIdx.z * blockDim.z + threadIdx.z;                              // 计算全局 z

    if (x < width && y < height && z < depth) {                                 // 边界检查
        int idx = z * (height * width) + y * width + x;                         // 计算线性索引
        tensor[idx] = z * 10000.0f + y * 100.0f + x;                            // 根据坐标赋值
    }                                                                           // if 结束
}                                                                               // tensor3d_process 结束

void run_3d_demo()                                                              // 定义运行 3D 示例的辅助函数
{
    printf("\n--- 3D 索引示例 ---\n");                                          // 打印标题

    dim3 block3(2, 2, 2);                                                       // 定义 2x2x2 线程块
    dim3 grid3(2, 2, 2);                                                        // 定义 2x2x2 网格
    printf("配置：grid %dx%dx%d, block %dx%dx%d\n", grid3.x, grid3.y, grid3.z, block3.x, block3.y, block3.z); // 打印配置
    print_3d_index<<<grid3, block3>>>();                                        // 启动 3D 打印核函数
    cudaDeviceSynchronize();                                                    // 等待执行完成

    printf("\n--- 3D 张量处理示例 ---\n");                                      // 打印 3D 张量标题
    int width = 5;                                                              // 张量宽度
    int height = 4;                                                             // 张量高度
    int depth = 3;                                                              // 张量深度
    int size = width * height * depth;                                          // 元素总数

    float *h_tensor = (float*)malloc(size * sizeof(float));                     // 主机端分配张量内存
    float *d_tensor;                                                            // 设备端指针
    cudaMalloc((void**)&d_tensor, size * sizeof(float));                        // 设备端分配张量内存

    dim3 block_t(2, 2, 2);                                                      // 线程块尺寸
    dim3 grid_t((width  + block_t.x - 1) / block_t.x,
                (height + block_t.y - 1) / block_t.y,
                (depth  + block_t.z - 1) / block_t.z);                          // 网格尺寸向上取整

    printf("张量大小：%d x %d x %d\n", width, height, depth);                   // 打印张量大小
    printf("launch grid: %d x %d x %d, block: %d x %d x %d\n",
           grid_t.x, grid_t.y, grid_t.z, block_t.x, block_t.y, block_t.z);      // 打印启动配置

    tensor3d_process<<<grid_t, block_t>>>(d_tensor, width, height, depth);      // 启动张量处理核函数
    cudaDeviceSynchronize();                                                    // 等待执行完成

    cudaMemcpy(h_tensor, d_tensor, size * sizeof(float), cudaMemcpyDeviceToHost); // 将结果拷回主机

    for (int z = 0; z < depth; ++z) {                                           // 遍历每个 z 切片
        printf("\nSlice z=%d:\n", z);                                           // 打印当前切片编号
        for (int y = 0; y < height; ++y) {                                      // 遍历 y
            for (int x = 0; x < width; ++x) {                                   // 遍历 x
                int idx = z * (height * width) + y * width + x;                 // 计算线性索引
                printf("%8.1f ", h_tensor[idx]);                                // 打印元素
            }                                                                   // x 循环结束
            printf("\n");                                                       // 换行
        }                                                                       // y 循环结束
    }                                                                           // z 循环结束

    cudaFree(d_tensor);                                                         // 释放设备端内存
    free(h_tensor);                                                             // 释放主机端内存

    printf("\n3D 示例完成。\n");                                                // 打印完成信息
}                                                                               // run_3d_demo 结束
__global__ void print_1d_index()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("1D: Block %d, Thread %d -> Global ID %d\n", 
           blockIdx.x, threadIdx.x, tid);
}

// 二维线程索引示例
__global__ void print_2d_index()
{
    // 计算二维全局索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算线性索引（假设是行主序）
    int tid = row * gridDim.x * blockDim.x + col;
    
    printf("2D: Block(%d,%d), Thread(%d,%d) -> Global(%d,%d) -> Linear %d\n",
           blockIdx.x, blockIdx.y, 
           threadIdx.x, threadIdx.y,
           col, row, tid);
}

// 矩阵元素处理示例
__global__ void matrix_process(float *matrix, int width, int height)
{
    // 计算当前线程处理的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 检查边界
    if (row < height && col < width) {
        int index = row * width + col;
        matrix[index] = row * 100.0f + col;  // 简单的赋值操作
    }
}

int main()
{
    printf("=== CUDA 线程索引详解 ===\n\n");
    
    // ========== 一维索引示例 ==========
    printf("--- 一维索引示例 ---\n");
    printf("配置：2 个线程块，每个块 3 个线程\n\n");
    print_1d_index<<<2, 3>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    // ========== 二维索引示例 ==========
    printf("--- 二维索引示例 ---\n");
    printf("配置：2x2 网格，每个块 2x2 线程\n\n");
    
    // 定义二维网格和线程块
    dim3 grid_size(2, 2);      // 2x2 的网格（4个线程块）
    dim3 block_size(2, 2);     // 2x2 的线程块（每个块4个线程）
    
    print_2d_index<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    // ========== 矩阵处理示例 ==========
    printf("--- 矩阵处理示例 ---\n");
    int width = 5;
    int height = 4;
    int size = width * height;
    
    // 分配主机内存
    float *h_matrix = (float*)malloc(size * sizeof(float));
    
    // 分配设备内存
    float *d_matrix;
    cudaMalloc((void**)&d_matrix, size * sizeof(float));
    
    // 配置线程块大小（通常选择 16x16 或 32x32）
    int block_width = 16;
    int block_height = 16;
    
    // 计算网格大小（向上取整）
    dim3 grid((width + block_width - 1) / block_width,
              (height + block_height - 1) / block_height);
    dim3 block(block_width, block_height);
    
    printf("矩阵大小：%d x %d\n", width, height);
    printf("网格大小：%d x %d 线程块\n", grid.x, grid.y);
    printf("线程块大小：%d x %d 线程\n", block.x, block.y);
    
    // 启动内核
    matrix_process<<<grid, block>>>(d_matrix, width, height);
    cudaDeviceSynchronize();
    
    // 复制结果回主机
    cudaMemcpy(h_matrix, d_matrix, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 打印矩阵
    printf("\n结果矩阵：\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%6.1f ", h_matrix[i * width + j]);
        }
        printf("\n");
    }
    
    // 清理
    cudaFree(d_matrix);
    free(h_matrix);
    
    printf("\n完成！\n");
    
    return 0;
}

/*
 * 编译和运行：
 * nvcc 03_thread_indexing.cu -o thread_indexing
 * ./thread_indexing
 * 
 * 关键概念：
 * 1. threadIdx: 三维向量 (x, y, z)，线程在线程块中的位置
 * 2. blockIdx: 三维向量 (x, y, z)，线程块在网格中的位置
 * 3. blockDim: 三维向量，线程块的维度（线程数）
 * 4. gridDim: 三维向量，网格的维度（线程块数）
 * 
 * 一维全局索引：
 *   tid = blockIdx.x * blockDim.x + threadIdx.x
 * 
 * 二维全局索引：
 *   row = blockIdx.y * blockDim.y + threadIdx.y
 *   col = blockIdx.x * blockDim.x + threadIdx.x
 * 
 * 线性索引（行主序）：
 *   index = row * width + col
 */

// ========== 3D 索引与 3D 张量示例 ==========

// 打印 3D 线程/块索引并给出线性全局索引
__global__ void print_3d_index()
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gz = blockIdx.z * blockDim.z + threadIdx.z;

    int total_x = gridDim.x * blockDim.x;
    int total_y = gridDim.y * blockDim.y;
    // 线性索引（按 z 主序，随后 y，最后 x）
    int linear = gz * (total_y * total_x) + gy * total_x + gx;

    printf("3D: Block(%d,%d,%d) Thread(%d,%d,%d) -> Global(%d,%d,%d) -> Linear %d\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           gx, gy, gz, linear);
}

// 简单的 3D 张量处理内核：将每个元素赋成 z*10000 + y*100 + x
__global__ void tensor3d_process(float *tensor, int width, int height, int depth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        int idx = z * (height * width) + y * width + x; // 行主序（z, y, x）
        tensor[idx] = z * 10000.0f + y * 100.0f + x;
    }
}

// 辅助函数：在 main 中调用此函数以运行 3D 示例
void run_3d_demo()
{
    printf("\n--- 3D 索引示例 ---\n");

    // 配置：小规格便于打印
    dim3 block3(2, 2, 2); // 每个线程块 2x2x2 = 8 线程
    dim3 grid3(2, 2, 2);  // 网格 2x2x2 = 8 块
    printf("配置：grid %dx%dx%d, block %dx%dx%d\n", grid3.x, grid3.y, grid3.z, block3.x, block3.y, block3.z);
    print_3d_index<<<grid3, block3>>>();
    cudaDeviceSynchronize();

    // ========== 3D 张量处理示例 ==========
    printf("\n--- 3D 张量处理示例 ---\n");
    int width = 5;
    int height = 4;
    int depth = 3;
    int size = width * height * depth;

    float *h_tensor = (float*)malloc(size * sizeof(float));
    float *d_tensor;
    cudaMalloc((void**)&d_tensor, size * sizeof(float));

    // 使用同样的 block 大小，计算合适的 grid（向上取整）
    dim3 block_t(2, 2, 2);
    dim3 grid_t((width  + block_t.x - 1) / block_t.x,
                (height + block_t.y - 1) / block_t.y,
                (depth  + block_t.z - 1) / block_t.z);

    printf("张量大小：%d x %d x %d\n", width, height, depth);
    printf("launch grid: %d x %d x %d, block: %d x %d x %d\n",
           grid_t.x, grid_t.y, grid_t.z, block_t.x, block_t.y, block_t.z);

    tensor3d_process<<<grid_t, block_t>>>(d_tensor, width, height, depth);
    cudaDeviceSynchronize();

    cudaMemcpy(h_tensor, d_tensor, size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印每个 z 平面
    for (int z = 0; z < depth; ++z) {
        printf("\nSlice z=%d:\n", z);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = z * (height * width) + y * width + x;
                printf("%8.1f ", h_tensor[idx]);
            }
            printf("\n");
        }
    }

    cudaFree(d_tensor);
    free(h_tensor);

    printf("\n3D 示例完成。\n");
}