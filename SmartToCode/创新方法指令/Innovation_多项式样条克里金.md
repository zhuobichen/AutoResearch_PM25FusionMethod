# 创新方法指令

## 方法名称
Polynomial Spline Kriging (PSK) - 多项式样条克里金

## 创新点
在 PolyRK 的二次多项式 OLS 全局校正 + 局部 GPR 克里金混合架构基础上，用三次样条（cubic spline）替代二次多项式进行全局趋势建模。多项式在边界区域容易出现剧烈振荡（Runge 现象），而三次样条在保持局部灵活性的同时保证全局光滑性，能更准确地捕捉 CMAQ 与监测值之间的非线性偏差关系。

## 背景问题

### 多项式的 Runge 现象
高阶多项式在插值区间边缘会出现剧烈振荡。二次多项式虽不严重，但在 CMAQ 与监测值的非线性关系较强时（如低浓度区斜率不同、高浓度区饱和效应），全局二次函数难以同时准确描述。

### 三次样条的优势
- **局部支撑性**：节点附近的影响更局部化
- **全局光滑性**：二阶连续可导，无多项式振荡
- **灵活节点**：可根据数据分布自适应放置节点

## 核心公式

### 全局三次样条校正（替代多项式 OLS）
$$
M_{cal}(s) = \sum_{k=0}^{K} \alpha_k \cdot B_k(CMAQ(s))
$$
其中 $B_k$ 为 B 样条基函数，$K$ 为节点数。

**自然边界条件**：
$$
\sum_{k} \alpha_k B_k(CMAQ) = \beta_0 + \beta_1 \cdot CMAQ + \beta_2 \cdot CMAQ^2 + \beta_3 \cdot CMAQ^3 + \sum_{j=1}^{K-2} \gamma_j (CMAQ - \kappa_j)^3_+
$$

### 残差计算
$$
R(s_i) = O(s_i) - M_{cal}(s_i)
$$

### GPR 残差克里金（与 PolyRK 相同）
$$
R^*(s) \sim GP(0, k_{RBF}(s, s'; \ell, \sigma))
$$

### 最终融合
$$
P_{PSK}(s) = M_{cal}(s) + R^*(s)
$$

## 关键步骤

### Step 1: 三次样条拟合
```
输入: m_train (CMAQ 站点值), y_train (监测值), n_knots (节点数)
处理:
  1. 计算节点位置（等间距或分位数）:
     knots = np.percentile(m_train, np.linspace(5, 95, n_knots))
  2. 构建三次样条基函数（BSpline 或 UnivariateSpline）:
     from scipy.interpolate import BSpline
     # 或者使用 saoverning SplineRegression
  3. 拟合样条系数: 最小二乘求解 alpha_k
  4. 计算训练残差: residual = y_train - spline.predict(m_train)
输出: 样条模型, residual 数组
```

**节点数选择策略**：
- 数据点 n < 50：K = 4-6 个节点
- 数据点 n = 50-150：K = 6-8 个节点
- 数据点 n > 150：K = 8-12 个节点
- 也可通过交叉验证选择最优 K

### Step 2: GPR 克里金（与 PolyRK 完全相同）
```
输入: X_train (lon, lat), residual
处理:
  1. kernel = ConstantKernel * RBF(length_scale=1.0) + WhiteKernel
  2. gpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=2, alpha=0.1)
  3. gpr.fit(X_train, residual)
  4. R_pred = gpr.predict(X_test)
输出: GPR 残差预测
```

### Step 3: 融合输出
```
输入: M_cal(X_test), R_pred
处理:
  P = M_cal(X_test) + R_pred
输出: PSK 融合预测
```

## 参数清单

| 参数符号 | 物理意义 | 取值范围 | 建议初值 |
|---------|---------|---------|---------|
| n_knots | 样条节点数 | 3-15 | 6 |
| spline_order | 样条阶数（3=三次） | 2-5 | 3 |
| spline_smooth | 光滑性惩罚因子 | 0.0-10.0 | 自动 |
| GPR length_scale | GPR 长度尺度 (km) | 5-50 | 15.0 |
| GPR alpha | GPR 正则化 | 0.01-1.0 | 0.1 |
| n_restarts | GPR 优化重启 | 1-5 | 2 |

## 与 PolyRK 的差异

| 方面 | PolyRK | PSK |
|------|--------|-----|
| 全局趋势 | 二次多项式（全局固定公式） | 三次样条（局部灵活） |
| 非线性建模 | 全局单一曲线 | 分段多项式拼接 |
| 边界行为 | 边缘振荡（Runge）风险 | 自然边界条件，无振荡 |
| 节点适应性 | 无节点概念 | 可根据 CMAQ 分布自适应节点 |

**核心差异**：PolyRK 用单一全局多项式描述 CMAQ-监测值的非线性关系；PSK 用分段三次多项式（样条）描述，允许不同 CMAQ 浓度区间有不同的偏差斜率，更符合实际物理（如饱和效应）。

## 预期效果
- R² >= 0.86（比 PolyRK 的 0.8519 提升约 0.01）
- MAE <= 6.9, RMSE <= 10.7
- 特别在高 CMAQ 浓度区域（多项式易饱和，样条更灵活）

## 方法指纹
MD5: `polynomial_spline_kriging_v1`

## 禁止依赖
- 不依赖气象数据（风速、温度、稳定度等）
- 不依赖时间数据（日期、季节等）
- 仅依赖 CMAQ 格点数据 + 监测站点坐标

## scipy 实现参考
```python
from scipy.interpolate import UnivariateSpline, BSpline
from sklearn.preprocessing import PolynomialFeatures

# 方法 A: UnivariateSpline（光滑样条）
spline = UnivariateSpline(m_train, y_train, k=3, s=None)  # s=光滑因子
residual = y_train - spline(m_train)

# 方法 B: BSpline（更灵活，节点可控）
from scipy.interpolate import make_interp_spline
knots = np.percentile(m_train, np.linspace(10, 90, 6))
coeffs, degree = fit_spline(m_train, y_train, k=3)  # 自定义拟合

# GPR 克里金（与 PolyRK 完全相同）
gpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=2, alpha=0.1)
gpr.fit(X_train, residual)
R_pred = gpr.predict(X_test)
```
