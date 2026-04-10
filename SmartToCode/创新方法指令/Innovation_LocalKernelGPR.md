# 创新方法指令

## 方法名称
Local Bandwidth Gaussian Process Regression (LB-GPR) - 局部带宽高斯过程回归

## 创新点
使用局部自适应带宽的 GPR 替代全局 GPR。传统 GPR 使用全局相关长度，但 CMAQ 偏差在空间上呈现非平稳性——城市中心相关长度短，郊区相关长度长。局部带宽 GPR 根据邻域站点密度和数据局部方差自适应调整核宽度，提高局地预测精度。

## 核心公式

### 局部自适应核函数
$$
k_{local}(s_i, s_j) = \sigma^2_f \cdot \exp\left(-\frac{1}{2} \sum_{d=1}^{2} \left(\frac{h_d}{\ell_d(s_i,s_j)}\right)^2\right) + \sigma^2_n \cdot \delta_{ij}
$$

### 局部相关长度（自适应带宽）
$$
\ell_d(s_i, s_j) = \sqrt{\frac{\ell_{d,min} + \ell_{d,max}}{2} + \beta \cdot |h_d|}
$$
或者使用站点密度估计：
$$
\ell(s) = \ell_0 \cdot \sqrt{\frac{n_{local}(s)}{\pi \cdot \rho_{target}}}
$$
- $n_{local}(s)$: 邻域站点数
- $\rho_{target}$: 目标密度参数

### 局地方差加权似然
$$
L_{local} = -\frac{1}{2} \sum_i w_i \left[ \log(2\pi\sigma_i^2) + \frac{R_i^2}{\sigma_i^2} \right]
$$
其中 $w_i$ 和 $\sigma_i^2$ 基于到目标点的局部距离和邻域方差估计。

### 最终预测
$$
P_{LBGPR}(s_0) = CMAQ(s_0) + \mu_R(s_0) + \sigma_R(s_0) \cdot \epsilon
$$
其中 $\mu_R(s_0), \sigma_R(s_0)$ 由局部 GPR 后验给出。

## 关键步骤
1. **邻域构建**：对每个目标点，选取 $k$ 个最近监测站
2. **局部带宽估计**：
   - 计算邻域站点的局地方差 $\sigma^2_{local}$
   - 根据站点稀疏度调整 $\ell$
3. **局部 GPR 拟合**：仅用邻域数据拟合 GPR 超参数
4. **后验预测**：GPR 后验均值 + 方差
5. **融合输出**：$P = CMAQ + \hat{R}_{local}$

## 参数清单
- $\ell_{0}$: 基础相关长度 (km), default: 15.0
- $\ell_{min}$: 最小相关长度 (km), default: 5.0
- $\ell_{max}$: 最大相关长度 (km), default: 40.0
- $\sigma_f$: 信号方差, default: 1.0
- $\sigma_n$: 噪声方差, default: 0.1
- $n_{neighbor}$: 邻域站点数, default: 10
- $\beta$: 带宽增长参数, default: 0.5

## 预期效果
- 局部 GPR 比全局 GPR 更适应空间非平稳性
- 预期 R² >= 0.86（相比 RK-Poly 的 0.8519 提升 >= 0.01）
- MAE <= 6.9, RMSE <= 10.8

## 为什么能超越 RK-Poly
RK-Poly 的残差克里金使用全局变异函数，假设空间平稳。LB-GPR 的创新：
1. 局部带宽自适应捕捉空间非平稳性
2. 城市高密度区短相关，郊区低密度区长三角
3. 仅用 Lat, Lon, CMAQ, Conc，无气象依赖

## 方法指纹
MD5: `local_bandwidth_gpr_v1`

## 创新判定
- 是否使用权重学习（Ridge/Lasso/线性回归）？否
- 是否有物理可解释性？是（局地相关长度反映局地扩散尺度）
- 是否依赖气象或时间数据？否
- 创新状态：保留
