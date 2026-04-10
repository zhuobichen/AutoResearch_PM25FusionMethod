# 【可执行方法规范】

## 方法名称
神经动态扩散-平流场与证据融合 (NeuroDDAF: Neural Dynamic Diffusion-Advection Fields with Evidential Fusion)

## 文献来源
- 论文标题：NeuroDDAF: Neural Dynamic Diffusion-Advection Fields with Evidential Fusion for Air Quality Forecasting
- 作者/年份：Dey et al. / 2026年
- 关键章节：Section 3 / Section 4

## 核心公式

**平流-扩散传输方程：**
$$
\frac{\partial c}{\partial t} = D \nabla^2 c - v \cdot \nabla c + S(t)
$$
其中 $c$ 为污染物浓度，$D$ 为扩散系数，$v$ 为风场，$S(t)$ 为源项。

**GRU时间编码：**
$$
h_t = GRU(x_t, h_{t-1})
$$

**风感知图注意力：**
$$
z_i = \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j \right)
$$
其中 $\alpha_{ij}$ 基于连接性和风特征。

**谱域PDE求解：**
$$
\hat{c}(k, t) = \mathcal{F}[c(x, t)]
$$
$$
\mathcal{F}[\nabla^2 c] = -\|k\|^2 \hat{c}(k, t)
$$
$$
\frac{\partial \hat{c}}{\partial t} = -D\|k\|^2 \hat{c} - i(v \cdot k)\hat{c} + R_\theta(\hat{c}, k, t)
$$

**证据门融合：**
$$
H_{fused} = \alpha \odot H_{diff} + (1-\alpha) \odot H_{adv}
$$
$$
\alpha = \sigma([H_{diff} \| H_{adv}] \phi(v) W_\alpha + b_\alpha)
$$

**不确定性估计：**
$$
V_{tot} = \frac{1}{S}\sum_{s=1}^{S} \Sigma^{(s)} + \frac{1}{S}\sum_{s=1}^{S}(\hat{Y}^{(s)} - \bar{Y})^2
$$

## 参数清单
| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| T | int | 24 | 序列长度 |
| τ | int | 24 | 预报步长 |
| d_ℓ | int | 4 | 隐变量维度 |
| GRU隐状态 | int | 64 | 隐向量大小 |
| K | int | 2 | 传播步数 |
| S | int | 3 | 随机轨迹数 |

## 数据规格
| 数据 | 格式 | 维度 |
|-----|------|-----|
| 污染物观测 | array | (N, T) |
| 气象数据 | array | (N, T, M) |
| 图结构 | graph | 监测网络 |
| 风场 | vector | (N, T, 2) |
| 预报输出 | (N, τ) | μg/m³ |

## 实现步骤
1. **GRU-GAT编码**：GRU捕获时间动态，风感知GAT捕获空间交互
2. **傅里叶域PDE求解**：在频域计算扩散-平流项，含可学习残差
3. **隐式Neural ODE积分**：用自适应Dormand-Prince求解器积分
4. **证据门融合**：自适应平衡物理引导和神经网络预测
5. **不确定性量化**：多轨迹采样计算总方差（认知+随机）

## 方法指纹
MD5: `neuroddaf_v1_physics_informed_diffusion_advection`

## 随机性
- [x] 是  - [ ] 否（MC dropout和随机轨迹）

## 备注
- 开系统动态建模（考虑跨边界交换）
- 物理可解释的扩散-平流机制
- 证据融合提供不确定性校准
- 北京数据集：RMSE 41.63 μg/m³ (1天) / 48.88 μg/m³ (3天)
