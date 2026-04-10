# 【可执行方法规范】

## 方法名称
AQ-Net - 时空神经网络法 (Deep Spatio-Temporal Neural Network for Air Quality)

## 文献来源
- 论文标题: "Deep Spatio-Temporal Neural Network for Air Quality Reanalysis"
- 作者: Ammar Kheder, Benjamin Foreback, Lili Wang, Zhi-Song Liu, Michael Boy
- 机构: Lappeenranta-Lahti University of Technology, University of Helsinki
- 日期: 2025
- arXiv: 2502.11941

## 核心公式

### 模型架构: AQ-Net

**输入层:**
- 历史污染物浓度 (t-T到t-1)
- 站点坐标 (经纬度)
- 气象数据（可选）

**时间建模: LSTM + Multi-Head Attention**
$$
h_t = LSTM(h_{t-1}, x_t)
$$
$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**空间建模: Neural kNN模块**
对每个目标位置，找到K个最近邻站点，使用神经网络权重融合:
$$
\hat{y}_{target} = \sum_{i=1}^{K} w_i \cdot y_i
$$
其中 $w_i = MLP(featurized\_distance_i)$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| hidden_dim | int | 128 | 隐藏层维度 |
| n_heads | int | 4 | 注意力头数 |
| n_layers | int | 2 | LSTM层数 |
| K | int | 10 | kNN近邻数 |
| seq_len | int | 168 | 输入序列长度（小时） |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 说明 |
|-----|------|-----|------|
| 历史浓度 | array | (n_station, seq_len, n_var) | 多变量时间序列 |
| 站点坐标 | array | (n_station, 2) | 经纬度 |
| 目标位置 | array | (n_target, 2) | 待预测位置 |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 预测浓度 | array | μg/m³ |

## 实现步骤

1. **LSTM时间建模**: 提取时间依赖特征
2. **注意力机制**: 识别关键时间步
3. **时间池化**: 聚合时间特征
4. **Neural kNN**: 基于学习的权重进行空间插值
5. **输出**: 生成目标位置的预测

## 方法优势
- 同时建模时间和空间依赖
- 注意力机制可解释
- 端到端训练

## 随机性
- [ ] 是（深度学习训练带随机初始化）

## 方法指纹
MD5: aqnet_spatiotemporal_nn

## 实现检查清单
- [ ] 核心公式已验证
- [ ] LSTM时间建模已实现
- [ ] Neural kNN已实现
