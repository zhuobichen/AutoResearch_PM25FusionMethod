# 创新方法指令

## 方法名称
CSP-RK-INT (CSP-RK with Interaction Terms) - 交互效应增强的浓度分层多项式残差克里金

## 创新点
在 CSP-RK 的浓度分层架构基础上，引入 CMAQ 与空间位置（经度、纬度）的交互项。传统 CSP-RK 只考虑 $O = f(M)$ 的关系，忽略了"同一 CMAQ 值在城市中心和郊区偏差可能不同"的空间异质性现象。本方法通过添加 $M \times Lat$ 和 $M \times Lon$ 交互项，允许偏差校正参数随空间位置变化，捕捉城市-郊区梯度效应。

## 背景问题

### CSP-RK 忽略空间异质性的问题
CSP-RK 对每个浓度层独立建模，但每个层内的 OLS 参数在整个空间域是统一的：
$$
O_i(s) = a_i + b_i M(s) + c_i M(s)^2 + \epsilon_i
$$

这假设同一 CMAQ 值在不同位置的偏差完全相同，忽略了：
1. **城市-郊区梯度**：城市中心建筑密集、人为源排放集中，同一 CMAQ 值在城市中心和郊区的真实浓度可能不同
2. **下垫面差异**：城市下垫面（不透水面）vs 自然下垫面（植被、水体）导致不同的混合层发展
3. **区域传输方向性**：主导风向不同导致上风向和下风向城市的偏差特征不同

### 交互效应的物理动机
引入 $M \times Lat$ 和 $M \times Lon$ 交互项，可以捕捉：
- 南北梯度：纬度越高，冬季供暖排放越多，同一 CMAQ 的真实浓度可能偏高
- 东西梯度：经度反映距海距离，沿海城市气溶胶湿沉降更强

## 核心公式

### 第一步：交互多项式特征
对每个浓度层，构建扩展特征：
$$
\mathbf{f}(M, Lat, Lon) = [1, M, M^2, M \cdot Lat, M \cdot Lon]
$$

### 第二步：分层交互多项式校正
$$
O(s) =
\begin{cases}
a_1 + b_1 M(s) + c_1 M(s)^2 + d_1 \cdot (M \cdot Lat) + e_1 \cdot (M \cdot Lon) + \epsilon_1 & M(s) < T_1 \\
a_2 + b_2 M(s) + c_2 M(s)^2 + d_2 \cdot (M \cdot Lat) + e_2 \cdot (M \cdot Lon) + \epsilon_2 & T_1 \leq M(s) < T_2 \\
a_3 + b_3 M(s) + c_3 M(s)^2 + d_3 \cdot (M \cdot Lat) + e_3 \cdot (M \cdot Lon) + \epsilon_3 & M(s) \geq T_2
\end{cases}
$$

### 第三步：残差计算（每层独立）
$$
R_i = O_i(s_i) - M_{cal}(s_i) \quad \text{（对每层分别计算）}
$$

### 第四步：GPR 残差克里金（与 CSP-RK 相同）
$$
R^*(s) \sim GP(0, k_{RBF}(s, s'; \ell, \sigma))
$$

### 第五步：融合结果
$$
P_{CSP\text{-}RK\text{-}INT}(s) = M_{cal}(s) + R^*(s)
$$

## 关键步骤

### Step 1: 交互特征构建
```
输入: m_train, X_train (lon, lat)
处理:
  1. poly = PolynomialFeatures(degree=2, include_bias=False)
  2. m_poly = poly.fit_transform(m_train.reshape(-1,1))  # [M, M^2]
  3. 创建交互特征:
     f1 = m_train * X_train[:, 0]  # M * Lon
     f2 = m_train * X_train[:, 1]  # M * Lat
  4. 拼接: X_ext = np.column_stack([m_poly, f1, f2])
输出: X_ext (n, 5)
```

### Step 2: 分层交互多项式拟合
```
输入: m_train, y_train, X_train, layer, T1, T2
处理:
  1. 对每层 i ∈ {1,2,3}:
     - mask = layer == i
     - m_i = m_train[mask], y_i = y_train[mask], X_i = X_train[mask]
     - 构建交互特征 X_ext_i
     - ols_i = LinearRegression().fit(X_ext_i, y_i)
     - residual_i = y_i - ols_i.predict(X_ext_i)
  3. 全局拼接: residual = np.concatenate([residual_1, residual_2, residual_3])
输出: ols_1, ols_2, ols_3, residual
```

### Step 3: 预测时分层应用
```
输入: m_test, X_test, ols_1, ols_2, ols_3, T1, T2
处理:
  1. poly = PolynomialFeatures(degree=2, include_bias=False)
  2. m_poly = poly.transform(m_test.reshape(-1,1))
  3. 构建交互特征:
     f1 = m_test * X_test[:, 0]
     f2 = m_test * X_test[:, 1]
     X_ext_test = np.column_stack([m_poly, f1, f2])
  4. 根据 m_test 值选择对应层的 OLS 模型
  5. M_cal_test = 对应层模型的预测值
输出: M_cal_test
```

### Step 4: GPR 克里金（与 CSP-RK 完全相同）
```
输入: X_train (lon, lat), residual
处理:
  1. kernel = ConstantKernel * RBF(length_scale=1.0) + WhiteKernel
  2. gpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=2, alpha=0.1)
  3. gpr.fit(X_train, residual)
  4. R_pred = gpr.predict(X_test)
输出: GPR 残差预测
```

### Step 5: 融合输出
```
输入: M_cal_test, R_pred
处理:
  P = M_cal_test + R_pred
输出: CSP-RK-INT 融合预测
```

## 参数清单

| 参数符号 | 物理意义 | 取值范围 | 建议初值 |
|---------|---------|---------|---------|
| T1 | 低-中浓度阈值 | 20-50 μg/m³ | 35 |
| T2 | 中-高浓度阈值 | 50-120 μg/m³ | 75 |
| degree | 多项式阶数 | 2 | 2 |
| interaction | 是否使用交互项 | True/False | True |

## 与 CSP-RK 的差异

| 方面 | CSP-RK | CSP-RK-INT |
|------|--------|------------|
| 特征维度 | $[1, M, M^2]$ (3维) | $[1, M, M^2, M \cdot Lat, M \cdot Lon]$ (5维) |
| 空间异质性 | 无（参数全局统一） | 有（交互项允许参数随位置变化） |
| 捕捉效应 | 仅浓度依赖偏差 | 浓度依赖 + 空间位置梯度 |
| 参数数量（每层） | 3 (a,b,c) | 5 (a,b,c,d,e) |
| 总参数数量 | 9 + GPR | 15 + GPR |

**核心差异**：CSP-RK-INT 通过添加 $M \times Lat$ 和 $M \times Lon$ 交互项，允许偏差校正系数在空间上连续变化，捕捉"同一 CMAQ 值在城市中心和郊区偏差不同"的空间异质性现象。

## 预期效果
- R² >= 0.856（比 CSP-RK 的 0.8535 提升约 0.0025）
- MAE <= 6.9, RMSE <= 10.8
- 特别在城市-郊区梯度明显的数据区域

## 方法指纹
MD5: `csp_rk_interaction_v1`

## 禁止依赖
- 不依赖气象数据（风速、温度、稳定度等）
- 不依赖时间数据（日期、季节等）
- 仅依赖 CMAQ 格点数据 + 监测站点坐标

## sklearn 实现参考
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def build_interaction_features(m, X, poly):
    """
    构建带交互项的特征矩阵
    m: CMAQ 值 (n,)
    X: 位置坐标 (n, 2) - [Lon, Lat]
    poly: PolynomialFeatures 实例
    """
    m_poly = poly.fit_transform(m.reshape(-1, 1))  # [M, M^2]
    m_lon = m * X[:, 0]  # M * Lon
    m_lat = m * X[:, 1]  # M * Lat
    return np.column_stack([m_poly, m_lon, m_lat])

def csp_rk_int_fit(m_train, y_train, X_train, T1, T2):
    """CSP-RK-INT 训练"""
    poly = PolynomialFeatures(degree=2, include_bias=False)
    layers = np.where(m_train < T1, 0, np.where(m_train < T2, 1, 2))

    models = {}
    residuals = []

    for layer in range(3):
        mask = layers == layer
        if np.sum(mask) < 5:  # 需要更多样本（5个特征）
            continue
        m_layer = m_train[mask]
        y_layer = y_train[mask]
        X_layer = X_train[mask]

        # 构建交互特征
        X_ext = build_interaction_features(m_layer, X_layer, poly)
        ols = LinearRegression().fit(X_ext, y_layer)
        residual_layer = y_layer - ols.predict(X_ext)

        models[layer] = ols
        residuals.extend(residual_layer.tolist())

    residual = np.array(residuals)
    kernel = ConstantKernel(10.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=2, alpha=0.1)
    gpr.fit(X_train, residual)

    return models, gpr, poly

def csp_rk_int_predict(m_test, X_test, models, gpr, poly, T1, T2):
    """CSP-RK-INT 预测"""
    layers = np.where(m_test < T1, 0, np.where(m_test < T2, 1, 2))
    X_ext = build_interaction_features(m_test, X_test, poly)

    M_cal = np.zeros_like(m_test)
    for layer in range(3):
        if layer in models:
            mask = layers == layer
            if np.sum(mask) > 0:
                M_cal[mask] = models[layer].predict(X_ext[mask])

    R_pred = gpr.predict(X_test)
    return M_cal + R_pred
```

## 交叉验证策略
十折交叉验证时，保持阈值 T1=35, T2=75 固定（与 CSP-RK 对比时），确保验证公平性。交互项的引入主要提升空间异质性捕捉能力。
