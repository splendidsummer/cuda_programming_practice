# CUDA 编程入门教程

这是一套完整的 CUDA 编程入门教程，从基础概念到实际应用，帮助初学者系统地学习 CUDA 并行计算。

## 目录结构

```
.
├── 01_hello_cuda.cu          # 第1课：Hello CUDA（基础内核函数）
├── 02_vector_add.cu          # 第2课：向量加法（内存管理）
├── 03_thread_indexing.cu     # 第3课：线程索引详解
├── 04_matrix_multiply.cu     # 第4课：矩阵乘法
├── 05_shared_memory.cu       # 第5课：共享内存详解
├── 06_memory_management.cu   # 第6课：内存管理
├── 07_error_handling.cu      # 第7课：错误处理
├── 08_reduce.cu              # 第8课：Reduce（归约）操作
├── 09_all_reduce.cu          # 第9课：All Reduce（全局归约）操作
├── 10_gather.cu              # 第10课：Gather（收集）操作
├── 11_all_gather.cu          # 第11课：All Gather（全局收集）操作
└── README.md                 # 本文件
```

## 环境要求

- NVIDIA GPU（支持 CUDA）
- CUDA Toolkit（建议 11.0 或更高版本）
- GCC 编译器
- Linux 操作系统

## 安装 CUDA

### 检查 GPU 和 CUDA

```bash
# 检查 GPU
nvidia-smi

# 检查 CUDA 版本
nvcc --version
```

### 安装 CUDA Toolkit

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit

# 或者从 NVIDIA 官网下载安装包
# https://developer.nvidia.com/cuda-downloads
```

## 编译和运行

### 基本编译命令

```bash
# 编译单个文件
nvcc 01_hello_cuda.cu -o hello_cuda

# 运行
./hello_cuda
```

### 编译选项

```bash
# 启用优化
nvcc -O3 01_hello_cuda.cu -o hello_cuda

# 启用调试信息
nvcc -g -G -lineinfo 01_hello_cuda.cu -o hello_cuda

# 指定计算能力
nvcc -arch=sm_75 01_hello_cuda.cu -o hello_cuda
```

### 使用 Makefile

可以创建一个 Makefile 来简化编译过程：

```makefile
NVCC = nvcc
CFLAGS = -O3
TARGETS = hello_cuda vector_add thread_indexing matrix_multiply shared_memory memory_management error_handling

all: $(TARGETS)

%: %.cu
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean
```

然后使用：
```bash
make
make clean
```

## 课程内容

### 第1课：Hello CUDA
- 创建第一个 CUDA 程序
- 理解内核函数（`__global__`）
- 线程块和网格的概念
- 内核启动语法 `<<< >>>`

**关键概念：**
- `threadIdx.x`: 线程在线程块中的索引
- `blockIdx.x`: 线程块在网格中的索引
- `blockDim.x`: 线程块的维度
- `cudaDeviceSynchronize()`: 同步主机和设备

### 第2课：向量加法
- GPU 内存分配（`cudaMalloc`）
- 主机和设备之间的数据传输（`cudaMemcpy`）
- 并行计算的基本模式
- 内存管理

**关键概念：**
- `cudaMalloc`: 在 GPU 上分配内存
- `cudaMemcpy`: 在主机和设备之间复制数据
- `cudaFree`: 释放 GPU 内存
- 内存访问模式

### 第3课：线程索引详解
- 一维、二维、三维线程索引
- 全局索引计算
- 矩阵处理的索引映射
- 线程块配置

**关键概念：**
- `dim3`: 三维向量类型
- 索引计算公式
- 边界检查
- 线程块大小选择

### 第4课：矩阵乘法
- 矩阵乘法的并行实现
- 共享内存优化
- 性能对比
- 平铺（Tiling）技术

**关键概念：**
- 共享内存（`__shared__`）
- `__syncthreads()`: 线程同步
- 性能优化技巧
- 内存访问模式优化

### 第5课：共享内存详解
- 共享内存的使用场景
- 并行归约（Reduction）
- 向量点积实现
- 性能优化

**关键概念：**
- 共享内存的特点
- 归约算法
- 原子操作（`atomicAdd`）
- 内存层次结构

### 第6课：内存管理
- 全局内存
- 共享内存
- 常量内存
- 统一内存
- 内存使用统计

**关键概念：**
- 各种内存类型的特点
- 内存分配和释放
- 内存访问优化
- 统一内存（Unified Memory）

### 第7课：错误处理
- CUDA 错误检查
- 错误处理宏
- 常见错误类型
- 调试技巧

**关键概念：**
- 错误检查最佳实践
- `cudaGetLastError()`
- 错误处理宏
- 调试工具

### 第8课：Reduce（归约）操作
- 线程块内的归约
- 全局归约（跨线程块）
- 不同归约操作（求和、求最大值、求最小值）
- 优化的归约算法

**关键概念：**
- 两阶段归约
- 共享内存优化
- 并行归约算法
- Warp shuffle 指令

### 第9课：All Reduce（全局归约）操作
- 所有线程块执行 reduce 操作
- 将结果广播给所有线程块
- 多轮内核调用实现
- 使用原子操作实现

**关键概念：**
- 多阶段 All Reduce
- 线程块间通信
- 结果广播
- 性能优化

### 第10课：Gather（收集）操作
- 从多个位置收集数据
- 使用索引数组指定收集位置
- 不同的 Gather 模式
- 性能优化技巧

**关键概念：**
- 索引数组
- 边界检查
- 内存访问模式
- 共享内存优化

### 第11课：All Gather（全局收集）操作
- 从所有线程块收集数据
- 将收集的数据广播给所有线程块
- 多阶段实现
- 性能优化

**关键概念：**
- 多阶段 All Gather
- 数据收集和广播
- 全局数据共享
- 集合通信操作

## 学习路径

### 初学者路径
1. **第1课** - 理解 CUDA 基本概念
2. **第2课** - 学习内存管理
3. **第3课** - 深入理解线程索引
4. **第4课** - 实现实际应用（矩阵乘法）
5. **第5课** - 学习性能优化
6. **第6课** - 全面理解内存系统
7. **第7课** - 掌握错误处理和调试
8. **第8课** - 学习 Reduce 操作
9. **第9课** - 学习 All Reduce 操作
10. **第10课** - 学习 Gather 操作
11. **第11课** - 学习 All Gather 操作

### 实践建议
1. 逐个运行每个示例程序
2. 修改参数，观察结果变化
3. 尝试修改代码，实现自己的功能
4. 对比不同实现的性能
5. 阅读 CUDA 官方文档

## 常见问题

### 1. 编译错误：找不到 nvcc
**解决方案：** 确保 CUDA Toolkit 已正确安装，并将 CUDA 路径添加到 PATH 环境变量。

### 2. 运行时错误：未检测到 GPU
**解决方案：** 
- 检查 GPU 是否被识别：`nvidia-smi`
- 检查驱动程序是否正确安装
- 检查 CUDA 版本兼容性

### 3. 内存不足错误
**解决方案：** 
- 减少数据大小
- 检查 GPU 内存使用情况
- 使用统一内存（Unified Memory）

### 4. 内核执行失败
**解决方案：** 
- 检查线程块配置是否有效
- 检查设备计算能力
- 使用错误检查代码定位问题

## 进阶学习资源

1. **CUDA 官方文档**
   - https://docs.nvidia.com/cuda/

2. **CUDA C++ 编程指南**
   - https://docs.nvidia.com/cuda/cuda-c-programming-guide/

3. **CUDA 最佳实践指南**
   - https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

4. **NVIDIA 开发者博客**
   - https://developer.nvidia.com/blog/

5. **CUDA 示例代码**
   - CUDA Toolkit 安装目录下的 `samples/` 文件夹

## 性能优化建议

1. **内存访问优化**
   - 使用共享内存减少全局内存访问
   - 确保合并访问（Coalesced Access）
   - 避免内存碎片

2. **线程配置优化**
   - 选择合适的线程块大小（通常 128-256）
   - 避免线程块过大或过小
   - 考虑占用率（Occupancy）

3. **算法优化**
   - 使用平铺技术
   - 减少同步操作
   - 避免分支发散（Branch Divergence）

4. **工具使用**
   - 使用 `nvprof` 或 `nsys` 分析性能
   - 使用 `cuda-memcheck` 检查内存错误
   - 使用 Visual Profiler 可视化性能

## 许可证

本教程代码仅供学习使用。

## 贡献

欢迎提交问题报告和改进建议！

## 联系方式

如有问题，请查看 CUDA 官方文档或 NVIDIA 开发者论坛。

---

**祝学习愉快！**

