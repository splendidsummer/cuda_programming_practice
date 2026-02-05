# Mixture-of-Experts（MoE）与 GEGLU 激活机制 —— 完整 Markdown 笔记

## 1. GEGLU ≠ MoE
GEGLU 是 FFN 内部的激活结构，而 MoE 是多个 FFN 专家的组合。两者完全不同。

## 2. MoE（Mixture-of-Experts）
MoE 的核心思想：通过门控网络让每个 token 只激活少量专家 FFN，从而实现“大参数、小计算”。

（以下略，内容与上一条助手回应一致）
