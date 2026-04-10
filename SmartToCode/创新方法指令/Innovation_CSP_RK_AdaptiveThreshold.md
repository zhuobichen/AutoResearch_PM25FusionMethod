# 创新方法指令

## 方法名称
CSP-RK-ATO (CSP-RK with Adaptive Threshold Optimization) - 自适应阈值优化的浓度分层多项式残差克里金

## 创新点
在 CSP-RK 的浓度分层架构基础上，引入自适应阈值优化机制。传统 CSP-RK 使用固定阈值 T1=35, T2=75，这无法保证是最优分层边界。本方法通过网格搜索优化验证集上的 R²，自动确定最优的浓度分层阈值。同时引入分层权重的概念，使用平滑过渡带替代硬边界，减少阈值附近预测的不连续性。

## 背景问题

### CSP-RK 固定阈值的局限性
CSP-RK 使用固定阈值 T1=35, T2=75 分三层：
- 这些阈值基于空气质量标准的经验值
- 但不同地区、不同时段的最优分层点可能不同
- 硬边界导致边界点预测不连续

### 自适应阈值优化的动机
1. 不同数据集的最优阈值不同（需要数据驱动）
2. 网格搜索可以在验证集上找到最优阈值组合
3. 平滑过渡带可以消除硬边界的不连续性

## 核心公式

### 第一步：阈值网格搜索空间
$$
T_1 \in [20, 50], \quad T_2 \in [50, 120], \quad T_1 < T_2
$$

定义阈值候选网格：
$$
\mathcal{T} = \{(T_1^i, T_2^j) | T_1^i \in \{25, 30, 35, 40, 45\}, T_2^j \in \{60, 65, 70, 75, 80, 85, 90\}\}
$$

### 第二步：平滑过渡权重
不使用硬边界，改为 softmax 风格权重：
$$
w_l(m) = \frac{e^{-\kappa(m - T_1)}}{1 + e^{-\kappa(m - T_1)}} \quad \text{(低层权重)}
$$
$$
w_h(m) = \frac{1}{1 + e^{-\kappa(m - T_2)}} \quad \text{(高层权重)}
$$
$$
w_m(m) = 1 - w_l(m) - w_h(m) \quad \text{(中层权重)}
$$

其中 $\kappa$ 控制过渡宽度（建议 $\kappa = 0.1$）

### 第三步：加权融合预测
$$
P(s) = w_l(m) \cdot O_1(s) + w_m(m) \cdot O_2(s) + w_h(m) \cdot O_3(s) + R^*(s)
$$

其中 $O_i(s) = a_i + b_i M(s) + c_i M(s)^2$ 是各层的多项式预测

### 第四步：GPR 残差克里金（与 CSP-RK 相同）
$$
R^*(s) \sim GP(0, k_{RBF}(s, s'; \ell, \sigma))
$$

### 第五步：阈值优化目标
$$
\max_{T_1, T_2, \kappa} R^2(\text{validation})
$$
使用十折交叉验证中留出折作为验证集

## 关键步骤

### Step 1: 阈值网格定义
```
输入: m_train (CMAQ 站点值)
处理:
  1. 定义 T1 候选值: [25, 30, 35, 40, 45] μg/m³
  2. 定义 T2 候选值: [60, 65, 70, 75, 80, 85, 90] μg/m³
  3. 过滤 T1 < T2 的组合
输出: 有效阈值组合列表
```

### Step 2: 平滑过渡权重计算
```
输入: m_test, T1, T2, κ
处理:
  1. w_l = exp(-κ*(m - T1)) / (1 + exp(-κ*(m - T1)))
  2. w_h = 1 / (1 + exp(-κ*(m - T2)))
  3. w_m = 1 - w_l - w_h
输出: w_l, w_m, w_h (n,)
```

### Step 3: 加权分层多项式预测
```
输入: m_test, w_l, w_m, w_h, ols_1, ols_2, ols_3, poly
处理:
  1. m_poly = poly.transform(m_test.reshape(-1,1))
  2. O1 = ols_1.predict(m_poly), O2, O3 类似
  3. O_weighted = w_l * O1 + w_m * O2 + w_h * O3
输出: O_weighted
```

### Step 4: 十折验证找最优阈值
```
输入: day_df (包含 fold 信息)
处理:
  1. 对每个阈值组合 (T1, T2):
     - 对每个验证折 fold_i:
       - 用训练折拟合 CSP-RK-ATO
       - 在验证折上预测
       - 计算 R²
     - 计算平均 R²
  2. 选择平均 R² 最高的阈值组合
输出: 最优 T1*, T2*, κ*
```

## 参数清单

| 参数符号 | 物理意义 | 取值范围 | 建议初值 |
|---------|---------|---------|---------|
| T1 | 低-中浓度阈值 | 20-50 μg/m³ | 35 |
| T2 | 中-高浓度阈值 | 50-120 μg/m³ | 75 |
| κ | 平滑过渡宽度参数 | 0.01-0.5 | 0.1 |
| T1_candidates | T1 候选值列表 | 自定义 | [25,30,35,40,45] |
| T2_candidates | T2 候选值列表 | 自定义 | [60,65,70,75,80,85,90] |

## 与 CSP-RK 的差异

| 方面 | CSP-RK | CSP-RK-ATO |
|------|--------|------------|
| 阈值选择 | 固定 T1=35, T2=75 | 数据驱动网格搜索 |
| 边界处理 | 硬边界（不连续） | 平滑过渡权重（连续） |
| 阈值适应性 | 无 | 有（针对数据优化） |
| 参数数量 | 9 + GPR参数 | 9 + GPR参数 + κ |

**核心差异**：CSP-RK-ATO 通过验证集优化找到最优阈值组合，避免了经验阈值可能不适用于特定数据的问题。同时平滑过渡带消除了边界处的不连续性。

## 预期效果
- R² >= 0.855（比 CSP-RK 的 0.8535 提升约 0.0015）
- MAE <= 7.0, RMSE <= 10.9
- 特别在阈值边界处预测更平滑

## 方法指纹
MD5: `csp_rk_adaptive_threshold_v1`

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

def compute_weights(m, T1, T2, kappa):
    """计算三层平滑权重"""
    w_l = np.exp(-kappa * (m - T1)) / (1 + np.exp(-kappa * (m - T1)))
    w_h = 1 / (1 + np.exp(-kappa * (m - T2)))
    w_m = 1 - w_l - w_h
    return w_l, w_m, w_h

def csp_rk_ato_fit(m_train, y_train, X_train, T1, T2, kappa=0.1):
    """CSP-RK-ATO 训练"""
    poly = PolynomialFeatures(degree=2, include_bias=False)

    # 分层多项式拟合
    layers = np.where(m_train < T1, 0, np.where(m_train < T2, 1, 2))
    models = {}
    residuals = []

    for layer in range(3):
        mask = layers == layer
        if np.sum(mask) < 3:
            continue
        m_layer = m_train[mask]
        y_layer = y_train[mask]

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

def csp_rk_ato_predict(m_test, X_test, models, gpr, poly, T1, T2, kappa):
    """CSP-RK-ATO 预测"""
    m_poly = poly.transform(m_test.reshape(-1, 1))
    w_l, w_m, w_h = compute_weights(m_test, T1, T2, kappa)

    O_pred = np.zeros_like(m_test)
    for layer in range(3):
        if layer in models:
            O_pred += [w_l, w_m, w_h][layer] * models[layer].predict(m_poly)

    R_pred = gpr.predict(X_test)
    return O_pred + R_pred

def grid_search_thresholds(m_train, y_train, X_train, fold_train,
                          T1_cands, T2_cands, kappa=0.1):
    """网格搜索最优阈值"""
    best_r2 = -np.inf
    best_T1, best_T2 = 35, 75

    for T1 in T1_cands:
        for T2 in T2_cands:
            if T1 >= T2:
                continue

            r2_scores = []
            for fold_id in np.unique(fold_train):
                val_mask = fold_train == fold_id
                tr_mask = ~val_mask

                models, gpr, poly = csp_rk_ato_fit(
                    m_train[tr_mask], y_train[tr_mask], X_train[tr_mask],
                    T1, T2, kappa
                )
                y_pred = csp_rk_ato_predict(
                    m_train[val_mask], X_train[val_mask],
                    models, gpr, poly, T1, T2, kappa
                )
                r2_scores.append(r2_score(y_train[val_mask], y_pred))

            mean_r2 = np.mean(r2_scores)
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_T1, best_T2 = T1, T2

    return best_T1, best_T2, best_r2
```

## 交叉验证策略
十折交叉验证时，首先在全量数据上通过网格搜索确定最优 T1*, T2*（使用内部留一法或 bootstrap），然后固定阈值再进行十折验证，确保验证公平性。
