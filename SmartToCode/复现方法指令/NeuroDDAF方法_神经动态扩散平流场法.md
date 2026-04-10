# 复现方法指令

## 方法名称
NeuroDDAF - 神经动态扩散-平流场与证据融合法 (Neural Dynamic Diffusion-Advection Fields with Evidential Fusion)

## 输入数据
- 监测站坐标：shape (n, 2)
- CMAQ网格数据：shape (lat, lon, time)
- 气象数据（u/v风速、温度、边界层高度）：shape (lat, lon, time, n_met)
- 监测PM2.5：shape (n, time)
- 图结构（监测网络邻接关系）：shape (n, n)

## 输出数据
- 融合结果：shape (lat, lon, time)
- 不确定性场：shape (lat, lon, time)

## 核心公式

### 平流-扩散传输方程（物理核心）
$$
\frac{\partial c}{\partial t} = D \nabla^2 c - v \cdot \nabla c + S(t)
$$
其中 $c$为污染物浓度，$D$为扩散系数（大气扩散率），$v=(u,v)$为风场，$S(t)$为源项。

### GRU时间编码（时间动态）
$$
h_t = GRU(x_t, h_{t-1})
$$
捕获时间依赖特征。

### 风感知图注意力（空间传播）
$$
z_i = \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j \right)
$$
其中 $\alpha_{ij}$ 基于节点间连接性和风场特征。

### 谱域PDE求解（计算高效）
$$
\hat{c}(k, t) = \mathcal{F}[c(x, t)]
$$
$$
\mathcal{F}[\nabla^2 c] = -\|k\|^2 \hat{c}(k, t)
$$
$$
\frac{\partial \hat{c}}{\partial t} = -D\|k\|^2 \hat{c} - i(v \cdot k)\hat{c} + R_\theta(\hat{c}, k, t)
$$
其中 $R_\theta$ 为可学习残差项（神经网络预测）。

### 证据门融合（不确定性校准）
$$
H_{fused} = \alpha \odot H_{diff} + (1-\alpha) \odot H_{adv}
$$
$$
\alpha = \sigma([H_{diff} \| H_{adv}] \phi(v) W_\alpha + b_\alpha)
$$
自适应平衡扩散和平流预测。

### 不确定性估计（总方差分解）
$$
V_{tot} = \frac{1}{S}\sum_{s=1}^{S} \Sigma^{(s)} + \frac{1}{S}\sum_{s=1}^{S}(\hat{Y}^{(s)} - \bar{Y})^2
$$
认知不确定度（模型）+ 随机不确定度（数据）。

## 关键步骤
1. **GRU-GAT编码**：GRU捕获时间动态，风感知GAT捕获空间交互
2. **傅里叶域PDE求解**：在频域计算扩散-平流项，含可学习残差
3. **隐式Neural ODE积分**：用自适应Dormand-Prince求解器积分
4. **证据门融合**：自适应平衡物理引导和神经网络预测
5. **不确定性量化**：多轨迹采样计算总方差

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| T | int | 24 | 输入序列长度(hours) |
| τ | int | 24 | 预报步长(hours) |
| d_ℓ | int | 4 | 隐变量维度 |
| GRU_hidden | int | 64 | GRU隐状态大小 |
| K | int | 2 | 传播步数 |
| S | int | 3 | 随机轨迹数（MC dropout） |
| diffusion_coeff | float | 1.0 | 扩散系数初始值 |

## 与系统的适配

本方法将CMAQ视为物理初始猜，通过：
- 神经网络的平流-扩散修正（CMAQ偏差校正）
- 谱域计算保证计算效率
- 证据融合提供不确定性量化

## 创新判定
- 是否使用权重学习（Ridge/Lasso/线性回归）？否（神经网络优化，硬PDE约束）
- 是否有物理可解释性？是（平流-扩散方程物理机制清晰）
- 是否保留：保留（复现方案）

## 方法指纹
MD5: `neuroddaf_v1_physics_informed_diffusion_advection`

## 复现来源
- 文献分析员_NeuroDDAF神经动态扩散平流场法_20260409.md

## 随机性
- [x] 是（MC dropout和随机轨迹采样）

## 验证方法
- 十折CV对比ResidualKriging的RMSE
- 检验扩散系数D是否符合大气物理量级（0.1-10 m²/s）
- 不确定性校准性检验（预测区间覆盖率）
