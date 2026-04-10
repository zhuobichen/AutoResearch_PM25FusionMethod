# 创新方法指令

## 方法名称
CMAQ-Gradient-Anisotropic Residual Kriging (CGARK) - CMAQ 梯度各向异性残差克里金

## 创新点
在 PolyRK 的二次多项式 OLS 全局校正 + 局部 GPR 克里金混合架构基础上，添加 CMAQ 梯度方向引导的各向异性 GPR。计算 CMAQ 格点场的局地浓度梯度，污染扩散沿梯度方向（从高浓度区向低浓度区）具有更长相关长度。使用各向异性 RBF 核替代各向同性 RBF 核，使 GPR 克里金的空间相关性建模更符合 PM2.5 传输物理。

## 背景物理

### PM2.5 扩散的方向性
污染物从高浓度区向低浓度区扩散：
- **沿梯度方向**：风向下游浓度逐渐降低，相关长度长（平流主导）
- **垂直梯度方向**：等浓度线方向，相关长度短（湍流混合主导）

### 各向异性相关长度
$$
a(\theta) = a_{\min} + (a_{\max} - a_{\min}) \cdot |\cos(\theta - \theta_g)|^\alpha
$$
- $\theta_g$：局地 CMAQ 梯度方向角
- $\theta$：预测点与数据点连线方向
- $a_{\max}/a_{\min}$：各向异性比（建议 2.5:1）

## 核心公式

### 全局多项式校正（与 PolyRK 相同）
$$
M_{cal}(s) = \alpha_0 + \alpha_1 \cdot CMAQ(s) + \alpha_2 \cdot CMAQ(s)^2
$$

### CMAQ 梯度计算
$$
\nabla CMAQ(s_{grid}) = \left( \frac{\partial CMAQ}{\partial x}, \frac{\partial CMAQ}{\partial y} \right)
$$
使用格点中心差分：
$$
\frac{\partial CMAQ}{\partial x} \approx \frac{CMAQ_{i+1,j} - CMAQ_{i-1,j}}{2 \Delta x}
$$

### 梯度方向角
$$
\theta_g(s_{grid}) = \arctan2\left( \frac{\partial CMAQ}{\partial y}, \frac{\partial CMAQ}{\partial x} \right)
$$

### 各向异性 RBF 核
$$
k_{anisotropic}(s_i, s_j; \theta_g) = \sigma^2 \exp\left( -\frac{d_{ij}^2(\theta_g)}{2\ell^2} \right)
$$

**各向异性距离度量**：
$$
d_{ij}^2(\theta_g) = \frac{(h_x \cos\theta_g + h_y \sin\theta_g)^2}{\ell_{\parallel}^2} + \frac{(-h_x \sin\theta_g + h_y \cos\theta_g)^2}{\ell_{\perp}^2}
$$
- $\ell_{\parallel}$：沿梯度方向（平行）长度尺度
- $\ell_{\perp}$：垂直梯度方向长度尺度
- 各向异性比：$\ell_{\parallel} / \ell_{\perp} = 2.5$

### GPR 残差克里金（各向异性）
$$
R^*(s) \sim GP(0, k_{anisotropic}(s, s'; \theta_g, \ell_{\parallel}, \ell_{\perp}))
$$

### 最终融合
$$
P_{CGARK}(s) = M_{cal}(s) + R^*(s)
$$

## 关键步骤

### Step 1: 多项式 OLS 校正（与 PolyRK 完全相同）
```
输入: CMAQ 站点值 m_train, 监测值 y_train
处理:
  1. poly = PolynomialFeatures(degree=2, include_bias=False)
  2. m_train_poly = poly.fit_transform(m_train.reshape(-1,1))
  3. ols = LinearRegression().fit(m_train_poly, y_train)
  4. residual = y_train - ols.predict(m_train_poly)
输出: ols 模型参数, residual 数组
```

### Step 2: CMAQ 格点梯度计算
```
输入: CMAQ 格点数据 (lat, lon, PM25)
处理:
  1. 对每个格点 (i,j) 计算:
     dx = (lon[i,j+1] - lon[i,j-1]) / 2
     dy = (lat[i+1,j] - lat[i-1,j]) / 2
     dCMAQ_dx = (CMAQ[i,j+1] - CMAQ[i,j-1]) / (2*dx)
     dCMAQ_dy = (CMAQ[i+1,j] - CMAQ[i-1,j]) / (2*dy)
  2. theta_g = arctan2(dCMAQ_dy, dCMAQ_dx)
  3. 对站点位置插值获取站点处梯度角（IDW 或最近格点）
输出: 每个站点的梯度方向角 theta_g
```

### Step 3: 各向异性 GPR 拟合
```
输入: X_train (lon, lat), residual, theta_g_train
处理:
  1. 构建各向异性距离矩阵 D_ij:
     h_x = lon_j - lon_i
     h_y = lat_j - lat_i
     D_ij^2 = (h_x*cos(theta_g) + h_y*sin(theta_g))^2 / l_par^2
            + (-h_x*sin(theta_g) + h_y*cos(theta_g))^2 / l_perp^2
  2. 使用自定义核函数或手动计算协方差矩阵
  3. gpr = GaussianProcessRegressor(kernel, alpha=0.1)
  4. gpr.fit(X_train, residual)
输出: 各向异性 GPR 模型
```

**简化实现（使用 sklearn）：**
```python
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# 近似：用全局平均梯度方向构建各向异性核
# 或在预测时对每个点使用局部梯度方向角调整
kernel = ConstantKernel * AnisotropicRBF(length_scale=[15.0, 6.0]) + WhiteKernel
```

### Step 4: GPR 预测与融合
```
输入: X_test, gpr_model, theta_g_test
处理:
  1. R_pred = gpr_model.predict(X_test)
  2. P = M_cal(X_test) + R_pred
输出: CGARK 融合预测
```

## 参数清单

| 参数符号 | 物理意义 | 取值范围 | 建议初值 |
|---------|---------|---------|---------|
| $\ell_{\parallel}$ | 沿梯度方向长度尺度 (km) | 15-40 | 25.0 |
| $\ell_{\perp}$ | 垂直梯度方向长度尺度 (km) | 5-20 | 10.0 |
| 各向异性比 | $\ell_{\parallel}/\ell_{\perp}$ | 1.5-4.0 | 2.5 |
| GPR alpha | GPR 正则化 | 0.01-1.0 | 0.1 |
| n_restarts | GPR 优化重启 | 1-5 | 2 |
| gradient_interp | 梯度角插值方法 | IDW/最近邻 | 最近邻 |

## 与 PolyRK 的差异

| 方面 | PolyRK | CGARK |
|------|--------|-------|
| GPR 核函数 | 各向同性 RBF | 各向异性 RBF（梯度方向） |
| 空间相关性 | 单一长度尺度 | 方向依赖长度尺度 |
| 物理基础 | 无 | CMAQ 梯度反映扩散方向 |
| 残差插值 | 标量距离 | 方向感知距离 |

**核心差异**：PolyRK 的 GPR 使用各向同性 RBF核，假设空间相关性在所有方向相同；CGARK 利用 CMAQ 梯度方向使相关长度沿扩散方向更长，更符合 PM2.5 传输物理。

## 预期效果
- R² >= 0.86（比 PolyRK 的 0.8519 提升约 0.01）
- MAE <= 6.8, RMSE <= 10.6
- 特别在风速较大、梯度明显的区域效果更佳

## 方法指纹
MD5: `cmaq_gradient_anisotropic_residual_kriging_v1`

## 禁止依赖
- 不依赖气象数据（风速、温度、稳定度等）
- 不依赖时间数据（日期、季节等）
- 仅依赖 CMAQ 格点数据 + 监测站点坐标
