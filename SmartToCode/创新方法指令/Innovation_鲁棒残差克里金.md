# 创新方法指令

## 方法名称
Robust Residual Kriging (RRK) - 鲁棒残差克里金

## 创新点
在 PolyRK 的二次多项式 OLS 全局校正 + 局部 GPR 克里金混合架构基础上，用鲁棒回归（Huber 回归）替代普通 OLS 进行多项式偏差校正。OLS 对异常值极度敏感，而实际监测数据中常存在设备故障、极端污染事件等异常值，Huber 回归通过软化损失函数对异常值更具鲁棒性，使全局趋势建模更稳健。

## 背景问题

### OLS 异常值敏感性
普通最小二乘的损失函数为：
$$
L_{OLS} = \sum_i (y_i - \hat{y}_i)^2
$$
对所有残差同等平方惩罚，单个大异常值即可严重歪曲拟合结果。

### Huber 损失函数
$$
L_{Huber}(r) =
\begin{cases}
\frac{1}{2} r^2 & \text{if } |r| \leq \delta \\
\delta |r| - \frac{1}{2} \delta^2 & \text{if } |r| > \delta
\end{cases}
$$
- 对于小残差（|r| <= δ）：与 OLS 相同（平方损失）
- 对于大残差（|r| > δ）：线性损失（降低异常值影响）

## 核心公式

### 全局多项式校正（Huber 回归替代 OLS）
$$
M_{cal}(s) = \alpha_0 + \alpha_1 \cdot CMAQ(s) + \alpha_2 \cdot CMAQ(s)^2
$$
参数通过 Huber 回归求解，迭代加权最小二乘。

### 残差计算
$$
R(s_i) = O(s_i) - M_{cal}(s_i)
$$

### GPR 残差克里金（与 PolyRK 相同）
$$
R^*(s) \sim GP(0, k_{RBF}(s, s'; \ell, \sigma))
$$
克里金预测: $R^* = \mu_R(s) + k^T K^{-1} (R - \mu_R)$

### 最终融合
$$
P_{RRK}(s) = M_{cal}(s) + R^*(s)
$$

## 关键步骤

### Step 1: Huber 多项式回归
```
输入: m_train (CMAQ), y_train (监测值)
处理:
  1. poly = PolynomialFeatures(degree=2, include_bias=False)
  2. m_train_poly = poly.fit_transform(m_train.reshape(-1,1))
  3. huber = HuberRegressor(epsilon=1.35, max_iter=200)
  4. huber.fit(m_train_poly, y_train)
  5. residual = y_train - huber.predict(m_train_poly)
  6. (可选) 计算 Huber 加权权重用于诊断
输出: huber 模型参数, residual 数组
```

**epsilon 参数说明：**
- ε = 1.0：高鲁棒性（异常值多时使用）
- ε = 1.35：统计效率与鲁棒性平衡（MASS 库默认值）
- ε = 1.5：轻度鲁棒（仅预防极端异常值）

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
输出: RRK 融合预测
```

### Step 4: 异常值诊断（可选后处理）
```
输出:
  1. 计算标准化残差: r_i = (y_i - M_cal(s_i)) / sigma
  2. 标记 |r_i| > 2.5 的站点为潜在异常
  3. 可视化残差空间分布
```

## 参数清单

| 参数符号 | 物理意义 | 取值范围 | 建议初值 |
|---------|---------|---------|---------|
| degree | 多项式阶数 | 1-3 | 2 |
| epsilon | Huber 损失阈值 | 1.0-2.0 | 1.35 |
| alpha_huber | Huber 正则化强度 | 1e-5-1e-2 | 1e-3 |
| max_iter | Huber 迭代次数 | 100-500 | 200 |
| GPR length_scale | GPR 长度尺度 (km) | 5-50 | 15.0 |
| GPR alpha | GPR 正则化 | 0.01-1.0 | 0.1 |
| n_restarts | GPR 优化重启 | 1-5 | 2 |

## 与 PolyRK 的差异

| 方面 | PolyRK | RRK |
|------|--------|-----|
| 全局校正 | OLS 线性/二次多项式 | Huber 回归二次多项式 |
| 异常值处理 | 无（OLS 对异常值敏感） | Huber 损失函数软化影响 |
| 参数估计 | 解析解（直接矩阵运算） | 迭代加权最小二乘 |
| 稳定性 | 存在异常值时趋势偏移 | 异常值影响受限 |

**核心差异**：RRK 在全局趋势建模阶段就考虑异常值，使 M_cal(s) 更准确反映 CMAQ 与监测值的真实偏差关系，而非被异常污染事件或设备故障歪曲。

## 预期效果
- R² >= 0.855（比 PolyRK 的 0.8519 略有提升）
- 特别在存在明显异常值的数据场景下优势更显著
- MAE <= 7.0, RMSE <= 10.9

## 方法指纹
MD5: `robust_residual_kriging_v1`

## 禁止依赖
- 不依赖气象数据（风速、温度、稳定度等）
- 不依赖时间数据（日期、季节等）
- 仅依赖 CMAQ 格点数据 + 监测站点坐标

## sklearn 实现参考
```python
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures

# Huber 回归（解析多项式趋势）
huber = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=200)
huber.fit(X_poly, y)
residual = y - huber.predict(X_poly)

# GPR 克里金（与 PolyRK 完全相同）
gpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=2, alpha=0.1)
gpr.fit(X_train, residual)
R_pred = gpr.predict(X_test)
```
