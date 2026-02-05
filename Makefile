NVCC ?= nvcc                       # NVCC 可被覆盖的变量，默认为 nvcc（CUDA 编译器）
NVCCFLAGS ?= -O3 -std=c++14          # 编译器标志：-O3 优化，使用 C++14
GENCODES = -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86  # 生成代码：针对 compute_86 架构，生成 sm_86 和 compute_86 代码
LIBS = -lcublas                      # 链接库：cublas

TARGET = matmul                      # 可执行目标名称（保留以兼容旧习惯）
# 收集当前目录下的所有 .cu 源文件
SRCS := $(wildcard *.cu)             # 使用 wildcard 收集当前目录下所有 .cu 源文件
# 为每个 .cu 源文件生成一个同名可执行程序（避免把多个含 main 的 .cu 链接到一起）
PROGS := $(SRCS:.cu=)

.PHONY: all clean                    # 声明伪目标 all 和 clean，避免与同名文件冲突

all: $(PROGS)                        # 默认目标：构建所有可执行文件（每个 .cu 一个可执行文件）

# 为每个源文件单独生成可执行文件，避免多个 main 的链接冲突
%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:                                 # 清理目标：删除构建生成文件
	rm -f $(PROGS) *.o *.ptx *.cubin  # 删除所有生成的可执行文件、目标文件、PTX 和 cubin 文件

$(TARGET): $(SRCS)
    $(NVCC) $(NVCCFLAGS) $(GENCODES) $^ -o $@ $(LIBS)
