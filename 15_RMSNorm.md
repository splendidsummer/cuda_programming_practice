# RMSNorm (Root Mean Square Layer Normalization) 详解

在 RMSNorm 中，`weights` 代表**可学习的缩放参数 (Learnable Scaling Parameter)**，通常在论文和深度学习框架（如 PyTorch）中记作 $\gamma$ (Gamma) 或 $g$。

## 1. 数学公式 (Math Formula)

这段代码对应的完整数学公式如下：

$$ y_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i $$

展开 $\text{RMS}(x)$ 后：

$$ y_i = \left( \frac{x_i}{\sqrt{\frac{1}{N} \sum_{j=1}^N x_j^2 + \epsilon}} \right) \cdot \gamma_i $$

## 2. 代码与公式的对应关系

*   `output[i]` $\rightarrow$ $y_i$ (最终输出)
*   `input[i]` $\rightarrow$ $x_i$ (输入元素)
*   `inv_rms` $\rightarrow$ $\frac{1}{\sqrt{\frac{1}{N} \sum x^2 + \epsilon}}$ (RMS 的倒数)
*   **`weights[i]`** $\rightarrow$ **$\gamma_i$ (缩放参数)**

## 3. `weights` 的作用是什么？

1.  **恢复表达能力**: 归一化操作（除以 RMS）会强制把数据的尺度压缩到统一的标准（均方根为 1）。这虽然有助于训练稳定，但可能会限制神经网络的表达能力。
2.  **学习最佳尺度**: `weights` (即 $\gamma$) 允许模型自己学习每一层特征应该有多大的幅度。如果模型觉得某个特征很重要，它可以把对应的 $\gamma$ 学得很大；如果觉得不重要，可以学得很小。

在 PyTorch 的 `torch.nn.RMSNorm` 中，这个参数就是 `weight` 属性。
