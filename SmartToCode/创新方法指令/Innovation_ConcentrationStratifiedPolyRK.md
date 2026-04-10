# 创新方法指令

## 方法名称
Concentration-Stratified PolyRK (CSP-RK) - 浓度分层多项式残差克里金

## 创新点
在 PolyRK 的二次多项式 OLS 全局校正 + 局部 GPR 克里金混合架构基础上，引入浓度分层机制：将 CMAQ 数据分为高/中/低三个浓度区间，分别拟合独立的多项式校正参数。不同污染水平下的模型偏差特性存在显著差异（如高浓度时模型倾向于低估，饱和效应明显；低浓度时关系更线性），统一的多项式参数难以同时最优捕捉所有浓度区的偏差特征。

## 背景问题

### PolyRK 全局多项式的局限性
PolyRK 使用全局二次多项式 $O = a + b \cdot M + c \cdot M^2$ 进行偏差校正，参数 $(a, b, c)$ 对所有浓度区间一致。然而：

1. **高浓度区**：常出现饱和效应，CMAQ 预测值偏高，偏差与浓度呈非线性关系
2. **中浓度区**：偏差关系相对线性，可能存在轻微系统性偏移
3. **低浓度区**：监测噪声相对更大，偏差可能更随机

### 浓度分层校正的物理动机
- PM2.5 生成化学在大气边界层内是非线性过程
- 高浓度时化学转化和二次气溶胶贡献更重要
- 区域传输贡献在不同浓度水平也不同

## 核心公式

### 第一步：浓度分层
$$
T_1 < T_2: \text{低浓度} \leq T_1,\quad \text{中浓度} \in (T_1, T_2),\quad \text{高浓度} \geq T_2
$$

**阈值选择策略**：
- $T_1$ = CMAQ 数据25%分位数（约 30-40 ug/m³）
- $T_2$ = CMAQ 数据75%分位数（约 80-120 ug/m³）
- 或使用国家空气质量标准分级：$T_1 = 35, T_2 = 75$（优/良/污染分界）

### 第二步：分层多项式校正
$$
O(s) =
\begin{cases}
a_1 + b_1 M(s) + c_1 M(s)^2 + \epsilon_1 & M(s) < T_1 \quad \text{（低浓度）} \\
a_2 + b_2 M(s) + c_2 M(s)^2 + \epsilon_2 & T_1 \leq M(s) < T_2 \quad \text{（中浓度）} \\
a_3 + b_3 M(s) + c_3 M(s)^2 + \epsilon_3 & M(s) \geq T_2 \quad \text{（高浓度）}
\end{cases}
$$

**边界处理**：
- 使用平滑过渡带（transition band）避免边界处不连续
- 或使用 logit-type 权重软化边界：
$$
w_i = \frac{1}{1 + e^{-\kappa (M - T)}}
$$
其中 $\kappa$ 控制过渡宽度

### 第三步：残差计算
$$
R_i = O_i - M_{cal}(s_i) \quad \text{（对每层分别计算）}
$$

### 第四步：GPR 残差克里金（与 PolyRK 相同）
$$
R^*(s) \sim GP(0, k_{RBF}(s, s'; \ell, \sigma))
$$
使用统一的 GPR 核函数参数（残差的全局空间相关性不变）

### 第五步：融合结果
$$
P_{CSPRK}(s) = M_{cal}(s) + R^*(s)
$$

## 关键步骤

### Step 1: 浓度分层与阈值确定
```
输入: m_train (CMAQ 站点值)
处理:
  1. 计算 CMAQ 分位数: q25, q75 = np.percentile(m_train, [25, 75])
  2. 或使用固定阈值: T1=35, T2=75 (μg/m³)
  3. 创建分层标签: layer = 0 if m < T1 else (1 if m < T2 else 2)
  4. 计算每层样本数，确保每层至少有 10 个样本
输出: T1, T2, layer 标签
```

### Step 2: 分层多项式拟合
```
输入: m_train, y_train, layer, T1, T2
处理:
  1. poly = PolynomialFeatures(degree=2, include_bias=False)
  2. 对每层 i ∈ {1,2,3}:
     - m_i = m_train[layer == i]
     - y_i = y_train[layer == i]
     - m_i_poly = poly.fit_transform(m_i.reshape(-1,1))
     - ols_i = LinearRegression().fit(m_i_poly, y_i)
     - M_cal_i = ols_i.predict(m_i_poly)
     - residual_i = y_i - M_cal_i
  3. 全局拼接: residual = np.concatenate([residual_1, residual_2, residual_3])
输出: ols_1, ols_2, ols_3, residual
```

### Step 3: 预测时分层应用
```
输入: m_test, ols_1, ols_2, ols_3, T1, T2
处理:
  1. poly = PolynomialFeatures(degree=2, include_bias=False)
  2. m_test_poly = poly.fit_transform(m_test.reshape(-1,1))
  3. 对每个测试点根据 m_test 值选择对应层的 OLS 模型
  4. M_cal_test = 对应层模型的预测值
输出: M_cal_test
```

### Step 4: GPR 克里金（与 PolyRK 完全相同）
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
输出: CSP-RK 融合预测
```

## 参数清单

| 参数符号 | 物理意义 | 取值范围 | 建议初值 |
|---------|---------|---------|---------|
| T1 | 低-中浓度阈值 | 20-60 μg/m³ | 35 或 q25 |
| T2 | 中-高浓度阈值 | 60-150 μg/m³ | 75 或 q75 |
| degree | 多项式阶数 | 1-3 | 2 |
| smooth_boundary | 边界平滑宽度 | 0-20 μg/m³ | 5 |
| GPR length_scale | GPR 长度尺度 (km) | 5-50 | 15.0 |
| GPR alpha | GPR 正则化 | 0.01-1.0 | 0.1 |
| n_restarts | GPR 优化重启 | 1-5 | 2 |

## 与 PolyRK 的差异

| 方面 | PolyRK | CSP-RK |
|------|--------|--------|
| 多项式参数 | 全局统一 (a, b, c) | 分层独立 (a₁,b₁,c₁), (a₂,b₂,c₂), (a₃,b₃,c₃) |
| 浓度依赖性 | 无（所有浓度同一参数） | 有（高/中/低独立建模） |
| 饱和效应 | 单一二次项 | 每层独立二次项，允许高浓度更弯曲 |
| 边界处理 | 无 | 平滑过渡带避免不连续 |
| 参数数量 | 3 | 9（但每层数据量减少） |

**核心差异**：PolyRK 假设 CMAQ-监测值的偏差关系在整个浓度范围内由同一二次多项式描述；CSP-RK 允许不同浓度区间有不同的偏差特征，特别是高浓度区的饱和效应和低浓度区的随机噪声。

## 预期效果
- R² >= 0.86（比 PolyRK 的 0.8519 提升约 0.01）
- MAE <= 6.9, RMSE <= 10.8
- 特别在高浓度区域（饱和效应被独立建模，不再被全局平均稀释）

## 方法指纹
MD5: `concentration_stratified_polyrk_v1`

## 禁止依赖
- 不依赖气象数据（风速、温度、稳定度等）
- 不依赖时间数据（日期、季节等）
- 仅依赖 CMAQ 格点数据 + 监测站点坐标

## sklearn 实现参考
```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def csp_rk_fit(m_train, y_train, X_train, T1, T2):
    """CSP-RK 训练"""
    poly = PolynomialFeatures(degree=2, include_bias=False)
    layers = np.where(m_train < T1, 0, np.where(m_train < T2, 1, 2))

    models = {}
    residuals = []

    for layer in range(3):
        mask = layers == layer
        if np.sum(mask) < 3:
            continue
        m_layer = m_train[mask]
        y_layer = y_train[mask]
        X_layer = X_train[mask]

        m_poly = poly.fit_transform(m_layer.reshape(-1, 1))
        ols = LinearRegression().fit(m_poly, y_layer)
        residual_layer = y_layer - ols.predict(m_poly)

        models[layer] = ols
        residuals.extend(residual_layer.tolist())

    residual = np.array(residuals)
    kernel = ConstantKernel(10.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=2, alpha=0.1)
    gpr.fit(X_train, residual)

    return models, gpr, poly

def csp_rk_predict(m_test, X_test, models, gpr, poly, T1, T2):
    """CSP-RK 预测"""
    layers = np.where(m_test < T1, 0, np.where(m_test < T2, 1, 2))
    m_poly = poly.transform(m_test.reshape(-1, 1))

    M_cal = np.zeros_like(m_test)
    for layer in range(3):
        mask = layers == layer
        if np.sum(mask) == 0:
            continue
        M_cal[mask] = models[layer].predict(m_poly[mask])

    R_pred = gpr.predict(X_test)
    return M_cal + R_pred
```

## 交叉验证策略
十折交叉验证时，保持阈值 T1, T2 在所有折中使用相同值（在全量数据上确定），确保验证公平性。
