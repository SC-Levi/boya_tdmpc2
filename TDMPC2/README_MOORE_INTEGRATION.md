# Moore混合专家架构与TDMPC2集成

本文档详细说明了如何将Moore混合专家架构集成到TDMPC2框架中，包括实现细节、使用方法和调试指南。

## 1. 系统概述

### 1.1 核心思想

本项目将Moore的混合专家架构（Multiple Expert Mixture）集成到TDMPC2中，替换原有的可学习任务嵌入机制。新架构具有以下特点：

- **多专家混合**: 使用多个专家网络，每个专家学习不同的特征表示
- **正交化约束**: 专家输出正交化，确保不同专家捕获不同特征
- **基于任务的动态权重**: 为每个任务动态分配专家权重
- **可配置专家数量**: 灵活调整专家数量以平衡性能和计算开销

### 1.2 优势

- **更强的表达能力**: 多专家混合提供更丰富的任务表示
- **更好的泛化性**: 通过专家分解和组合，提高模型在不同任务上的泛化能力
- **更高效的训练**: 正交约束促进更高效的特征学习
- **可解释性**: 任务-专家权重提供了任务表示的可解释性

## 2. 核心组件

### 2.1 Moore层实现 (`moore_layers.py`)

实现了Moore混合专家架构的核心模块：

- `OrthogonalLayer1D`: 实现正交化约束的神经网络层
- `MooreExpertsModule`: 实现多专家网络架构
- `MooreTaskEncoder`: 实现任务ID到专家权重的映射
- `MooreTaskEmbeddingModule`: 集成专家网络和任务编码器的完整模块

### 2.2 Moore世界模型 (`moore_world_model.py`)

扩展原始世界模型，集成Moore任务嵌入模块：

- `MooreWorldModel`: 继承自`WorldModel`，替换原有任务嵌入机制

### 2.3 Moore强化学习代理 (`moore_tdmpc2.py`)

实现使用Moore世界模型的TDMPC2代理：

- `MooreTDMPC2`: 继承自`TDMPC2`，使用`MooreWorldModel`并添加专家统计功能

## 3. 使用指南

### 3.1 安装依赖

确保已安装以下依赖：
```bash
pip install torch numpy matplotlib
```

### 3.2 配置参数

Moore架构的关键配置参数：

- `n_experts`: 专家数量（默认：4）
- `temperature`: 专家权重softmax温度参数（默认：1.0）
- `use_softmax`: 是否使用softmax进行权重归一化（默认：True）
- `expert_hidden_dims`: 专家网络隐藏层维度（默认：[256, 256]）
- `debug_task_emb`: 是否启用调试输出（默认：False）

### 3.3 使用示例

替代标准TDMPC2，使用Moore版本：

```python
from tdmpc2.moore_tdmpc2 import MooreTDMPC2

# 创建配置
cfg = Config(...)  # 标准TDMPC2配置
cfg.n_experts = 4  # 设置Moore专家数量
cfg.temperature = 1.0  # 设置温度参数

# 创建Moore代理
agent = MooreTDMPC2(cfg)

# 使用与标准TDMPC2相同的API
action = agent.act(obs, t0=True, eval_mode=False, task=0)

# 获取专家统计信息
expert_stats = agent.get_expert_stats()
```

### 3.4 运行测试

我们提供了多个测试脚本：

1. **基本集成测试**:
```bash
python moore_integration_test.py --mode basic
```

2. **可视化测试**:
```bash
python moore_integration_test.py --mode visual
```

3. **性能测试**:
```bash
python moore_integration_test.py --mode perf
```

4. **全面测试**:
```bash
python moore_integration_test.py --mode all
```

## 4. 调试与分析

### 4.1 开启调试输出

设置`cfg.debug_task_emb = True`以获取详细的任务嵌入信息和专家权重分布。

### 4.2 可视化工具

使用`moore_integration_test.py`的可视化功能：

- **专家权重分布**: 显示每个任务对专家的权重分配
- **嵌入相似度**: 分析不同任务嵌入之间的相似度
- **专家正交性**: 验证专家输出是否正交

### 4.3 性能分析

测试脚本提供的性能测试功能可以对比标准TDMPC2和不同专家数量的Moore版本在推理速度上的差异。

### 4.4 常见问题与解决方案

1. **推理速度慢**: 减少专家数量或简化专家网络架构
2. **任务嵌入质量不佳**: 检查正交化约束是否有效，增加温度参数值
3. **部分专家未被使用**: 降低温度参数值，促进更均匀的专家使用

## 5. 技术细节

### 5.1 专家网络设计

每个专家是一个多层感知机，其输出通过正交化层处理，确保专家间的正交性：

```
Input -> Linear -> ReLU -> Linear -> ... -> Linear -> Orthogonalization -> Output
```

### 5.2 权重计算

任务专家权重计算过程：

1. 将任务ID嵌入为特征向量
2. 通过全连接网络映射到专家数量的向量
3. 应用softmax归一化（可选）
4. 使用温度参数调整分布

### 5.3 正交化实现

使用批量Gram-Schmidt正交化算法，具体步骤：

1. 对第一个向量归一化
2. 对后续每个向量，减去其在前面向量上的投影
3. 归一化处理后的向量

## 6. 进阶功能

### 6.1 专家统计收集

`MooreTDMPC2`实现了专家使用统计收集功能：

- **专家使用频率**: 统计每个专家被激活的频率
- **专家权重分布**: 记录每个任务的平均专家权重
- **专家权重熵**: 计算权重分布的熵，评估权重集中度

### 6.2 熵正则化

可以通过在损失函数中添加熵项，鼓励更均匀的专家使用：

```python
# 计算专家权重熵
entropy = -torch.sum(weights * torch.log(weights + 1e-10), dim=1).mean()
# 添加到损失函数，鼓励更高的熵（更均匀的权重）
loss = original_loss - entropy_coef * entropy
```

## 7. 后续改进方向

1. **上下文感知权重**: 使权重依赖于观测而非仅依赖任务ID
2. **动态专家数量**: 实现自适应专家数量机制
3. **层次化专家结构**: 引入多层专家架构
4. **注意力机制**: 使用注意力机制替代简单的线性组合

## 8. 致谢

感谢Moore架构的原创者提供的混合专家思想，以及TDMPC2框架的开发者提供的强大基础。 