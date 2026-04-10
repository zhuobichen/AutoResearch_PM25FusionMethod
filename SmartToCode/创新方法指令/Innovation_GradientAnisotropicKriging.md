# 创新方法指令

## 方法名称
Gradient Anisotropic Residual Kriging (GARK) - 梯度各向异性残差克里金

## 创新点
利用 CMAQ 格点场计算局地浓度梯度方向，在梯度主轴方向上进行各向异性克里金插值，无需任何气象数据。传统克里金假设各向同性，但 PM2.5 扩散往往沿梯度方向（从高浓度区向低浓度区）具有更长相关长度。

## 核心公式

### 局地梯度计算（纯空间计算）
$$
\nabla CMAQ(s) = \left( \frac{\partial CMAQ}{\partial x}, \frac{\partial CMAQ}{\partial y} \right)
$$
使用 CMAQ 格点数值差分近似：
$$
\frac{\partial CMAQ}{\partial x} \approx \frac{CMAQ_{i+1,j} - CMAQ_{i-1,j}}{2 \Delta x}
$$

### 梯度方向角
$$
\theta_g(s) = \arctan\left( \frac{\partial CMAQ / \partial y}{\partial CMAQ / \partial x} \right)
$$

### 各向异性变异函数
$$
\gamma(h, \theta) = c_0 + c \cdot \left[1 - \exp\left(-\frac{h}{a(\theta)}\right)\right]
$$
其中各向异性相关长度：
$$
a(\theta) = a_{min} + (a_{max} - a_{min}) \cdot \left| \cos(\theta - \theta_g) \right|^\alpha
$$
- 沿梯度方向（$\theta = \theta_g$）：$a = a_{max}$（长相关）
- 垂直梯度方向（$\theta = \theta_g + \pi/2$）：$a = a_{min}$（短相关）

### 融合公式
$$
P_{GARK}(s_0) = CMAQ(s_0) + R^*(s_0)
$$
其中 $R^*$ 是各向异性克里金插值的残差。

## 关键步骤
1. **CMAQ 格点梯度计算**：对每个网格点，用中心差分计算 $\nabla CMAQ$
2. **主轴方向确定**：取梯度方向 $\theta_g$ 作为主轴
3. **各向异性变异函数拟合**：残差变异函数考虑方向性，$a_{max}/a_{min}$ 比率建议 2:1 到 3:1
4. **克里金权重求解**：使用各向异性距离度量
   $$
   d_{ij}^2 = \frac{(h_x \cos\theta_g + h_y \sin\theta_g)^2}{a_{max}^2} + \frac{(-h_x \sin\theta_g + h_y \cos\theta_g)^2}{a_{min}^2}
   $$
5. **残差插值与融合**：$P = CMAQ + R^*$

## 参数清单
- $a_{min}$: 垂直梯度方向相关长度 (km), default: 8.0
- $a_{max}$: 沿梯度方向相关长度 (km), default: 20.0
- $\alpha$: 各向异性指数, default: 2.0
- $c_0$: 块金效应, default: 0.05
- $c$: 基台值, default: 1.0
- $n_{neighbor}$: 克里金邻域站点数, default: 12

## 预期效果
- 各向异性建模比 RK-Poly 的各向同性假设更符合 PM2.5 传输物理
- 预期 R² >= 0.86（相比 RK-Poly 的 0.8519 提升 >= 0.01）
- MAE <= 6.9, RMSE <= 10.8

## 为什么能超越 RK-Poly
RK-Poly 使用二阶多项式校正 + 各向同性克里金残差。GARK 的创新在于：
1. CMAQ 梯度直接反映污染物扩散方向
2. 沿梯度方向的相关长度更长，物理上合理
3. 无需气象数据，仅用 CMAQ 格点计算梯度

## 方法指纹
MD5: `gradient_anisotropic_residual_kriging_v1`

## 创新判定
- 是否使用权重学习（Ridge/Lasso/线性回归）？否
- 是否有物理可解释性？是（PM2.5 沿浓度梯度扩散）
- 是否依赖气象或时间数据？否
- 创新状态：保留
