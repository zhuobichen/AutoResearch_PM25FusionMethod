# 方法名称
物理引导时空图注意力网络 (Physical-Guided Spatio-Temporal Graph Attention Network, PG-STGAT)

## 类型
创新

## 核心公式

### 1. CMAQ物理约束残差建模
$$
R(s,t) = Y_{obs}(s,t) - \underbrace{f_{CMAQ}(s,t)}_{\text{CMAQ物理模型输出}}
$$
将监测残差 $R(s,t)$ 作为待学习目标，物理模型提供确定性先验。

### 2. 站点图结构构建
$$
A_{ij} = \exp\left(-\frac{d_{ij}}{\sigma}\right) \cdot \exp\left(-\frac{\Delta t_{ij}}{\tau}\right)
$$
空间邻接权重由站点间距离 $d_{ij}$ 和时间间隔 $\Delta t_{ij}$ 共同决定，$\sigma$ 和 $\tau$ 为尺度参数。

### 3. 图注意力机制
$$
\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(a^T[W h_i || W h_j]))}{\sum_{k \in \mathcal{N}_i} \exp(\text{LeakyReLU}(a^T[W h_k || W h_j]))}
$$
注意力系数 $\alpha_{ij}$ 衡量站点 $j$ 对站点 $i$ 的影响权重，$[||]$ 表示向量拼接。

### 4. 风场传输方向约束
$$
w_{ij}^{wind} = \begin{cases}
\cos(\theta_{wind} - \theta_{ij}) & \text{顺风方向} \\
0 & \text{逆风方向}
\end{cases}
$$
气象风场方向 $\theta_{wind}$ 与站点连线方向 $\theta_{ij}$ 的夹角决定传输权重。

### 5. 时空特征融合
$$
h_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} \cdot w_{ij}^{wind} \cdot W^{(l)} h_j^{(l)} + b^{(l)}\right)
$$
融合空间邻域信息、风场权重和可学习变换。

## 算法步骤

1. **步骤1：物理约束残差计算**
   - 获取CMAQ模型网格输出 $f_{CMAQ}(s,t)$
   - 将CMAQ插值到监测站点位置
   - 计算监测残差 $R(s,t) = Y_{obs}(s,t) - f_{CMAQ}(s,t)$

2. **步骤2：时空图构建**
   - 基于KNN算法构建空间邻接图（k=10）
   - 计算站点间风场传输权重 $w_{ij}^{wind}$
   - 构建时序边权重（考虑时间衰减）

3. **步骤3：图注意力特征学习**
   - 多头注意力机制（head=4）提取空间依赖
   - 门控循环单元（GRU）建模时序动态
   - 残差连接保证物理一致性

4. **步骤4：融合预测输出**
   - $Y_{fusion}(s,t) = f_{CMAQ}(s,t) + \hat{R}(s,t)$
   - 叠加物理先验与学习残差
   - 生成公里级PM2.5融合网格

## 参数说明
- k_neighbors：空间K近邻数，默认10
- attention_heads：多头注意力头数，默认4
- hidden_dim：隐藏层维度，默认64
- gru_layers：GRU层数，默认2
- wind_decay：风场权重衰减系数，默认0.5
- learning_rate：学习率，默认0.001
- batch_size：批处理大小，默认32
- epochs：训练轮数，默认100

## 预期效果
- R2：0.82-0.88
- RMSE：10-15 μg/m³
- MB：±2 μg/m³（保留物理先验偏差校正）

## 验证方案
十折交叉验证：按地理区域划分十折，评估空间外推泛化能力，重点验证：
1. 有CMAQ覆盖区域的残差学习效果
2. 无CMAQ覆盖区域的风场传输预测能力
3. 极端污染事件（PM2.5>150 μg/m³）的捕捉能力
