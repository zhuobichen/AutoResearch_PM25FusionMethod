# 创新方法指令

## 方法名称
CSP-RK-HLG (CSP-RK with Hybrid Layered GPR) - 混合分层高斯过程残差克里金

## 创新点
在 CSP-RK 的浓度分层 OLS 架构基础上，引入混合分层 GPR 策略。传统 CSP-RK 使用全局 GPR 建模残差，忽略了不同浓度层残差的空间自相关特性可能不同。本方法先用分层 OLS 校正（成功），再用各层独立/半独立的 GPR 混合策略建模残差，通过 Matern 核增强空间自相关捕捉能力。避免 HGP-RK 的"每层单独 GPR 样本太少"问题，采用全局初始化+局部分层微调的混合策略。

## 背景问题

### HGP-RK 失败的原因分析
HGP-RK（每层独立 GPR）失败（R²=0.6994）是因为：
1. 分层后每层样本量太少（约 1/3 全量）
2. GPR 在小样本上容易过拟合
3. 空间长度尺度无法正确估计

### CSP-RK 成功的经验
CSP-RK（R²=0.8535）成功是因为：
1. 分层 OLS 校正有效捕捉了浓度依赖的偏差
2. 全局 GPR 仍有足够样本建模空间相关性
3. 分层OLS + 全局GPR 的混合策略平衡了样本量和复杂度

### 混合分层 GPR 的动机
残差的空间自相关特性在不同浓度层可能不同：
1. **高浓度层**：局地排放源影响大，空间变异更剧烈，长度尺度可能较小
2. **低浓度层**：区域背景贡献大，空间变异更平滑，长度尺度可能较大
3. **全局 GPR**：忽略这些差异，可能低估某些层的空间自相关

## 核心公式

### 第一步：分层多项式校正（与 CSP-RK 相同）
$$
O_i(s) = a_i + b_i M(s) + c_i M(s)^2 + \epsilon_i, \quad i \in \{1, 2, 3\}
$$

### 第二步：各层独立计算残差
$$
R_i = O_i(s_i) - M_{cal}(s_i)
$$

### 第三步：混合 GPR 策略

**策略 A：分层 Matern GPR**
各层使用独立的 Matern 核 GPR，但通过全局数据初始化长度尺度：
$$
R_i^*(s) \sim GP(0, k_{Matern}(s, s'; \ell_i, \sigma_i, \nu))
$$

**策略 B：全局-局部混合 GPR**
$$
R^*(s) = R_{global}^*(s) + \sum_{i=1}^{3} w_i(s) \cdot R_{local,i}^*(s)
$$
其中 $w_i(s)$ 是空间权重（基于测试点到各层样本的距离）

### 第四步：Matern 核函数
$$
k_{Matern}(r) = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu} r}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu} r}{\ell}\right)
$$

推荐 $\nu = 2.5$（三次 Matern，足够平滑）

### 第五步：融合结果
$$
P_{CSP\text{-}RK\text{-}HLG}(s) = M_{cal}(s) + R^*(s)
$$

## 关键步骤

### Step 1: 分层 OLS 校正（与 CSP-RK 相同）
```
输入: m_train, y_train, X_train, T1, T2
处理:
  1. poly = PolynomialFeatures(degree=2, include_bias=False)
  2. layers = np.where(m_train < T1, 0, np.where(m_train < T2, 1, 2))
  3. 对每层 i ∈ {1,2,3}:
     - mask = layers == i
     - m_i = m_train[mask], y_i = y_train[mask]
     - m_i_poly = poly.fit_transform(m_i.reshape(-1,1))
     - ols_i = LinearRegression().fit(m_i_poly, y_i)
     - residual_i = y_i - ols_i.predict(m_i_poly)
  4. 存储 ols_i 和 residual_i
输出: ols_1, ols_2, ols_3, residual_1, residual_2, residual_3, layer_models
```

### Step 2: 全局 GPR 初始化
```
输入: X_train, all_residuals
处理:
  1. kernel_global = ConstantKernel * RBF + WhiteKernel
  2. gpr_global = GaussianProcessRegressor(kernel, n_restarts_optimizer=2, alpha=0.1)
  3. gpr_global.fit(X_train, all_residuals)
  4. 提取优化后的长度_scale_global
输出: length_scale_global, gpr_global
```

### Step 3: 分层 Matern GPR 微调
```
输入: X_train, residual_i, length_scale_global, layer_id
处理:
  1. kernel_layer = ConstantKernel(1.0) * Matern(length_scale=length_scale_global, nu=2.5) + WhiteKernel(1.0)
  2. gpr_layer = GaussianProcessRegressor(kernel_layer, n_restarts_optimizer=1, alpha=0.5)
  3. gpr_layer.fit(X_train, residual_i)  # 使用全量 X，但只用当前层的残差
  4. 优化该层特有的长度尺度
输出: gpr_layer_i
```

### Step 4: 混合预测
```
输入: X_test, gpr_global, gpr_layer_1, gpr_layer_2, gpr_layer_3, layer_models
处理:
  1. M_cal = np.zeros(n)
  2. 对每个测试点:
     - 根据 m_test 值选择对应层的 ols_i
     - M_cal[i] = ols_i.predict(m_test_poly[i])
  3. R_global = gpr_global.predict(X_test)
  4. 如果使用分层微调:
     - R_layer_i = gpr_layer_i.predict(X_test)
     - R_pred = R_global + 0.3 * (R_layer - R_global)  # 微调比例
  5. 否则 R_pred = R_global
输出: P = M_cal + R_pred
```

## 参数清单

| 参数符号 | 物理意义 | 取值范围 | 建议初值 |
|---------|---------|---------|---------|
| T1 | 低-中浓度阈值 | 20-50 μg/m³ | 35 |
| T2 | 中-高浓度阈值 | 50-120 μg/m³ | 75 |
| nu | Matern 核参数 | 1.5, 2.5, 3.5 | 2.5 |
| alpha | GPR 正则化参数 | 0.1-1.0 | 0.1 (global), 0.5 (layer) |
| fine_tune_ratio | 分层微调比例 | 0.0-1.0 | 0.3 |
| length_scale_init | 长度尺度初值 | 5-50 km | 15.0 |

## 与 CSP-RK 的差异

| 方面 | CSP-RK | CSP-RK-HLG |
|------|--------|------------|
| GPR 核函数 | RBF only | Matern (nu=2.5) + 全局初始化 |
| 空间自相关 | 全局统一建模 | 分层独立 + 全局微调 |
| 核函数灵活性 | 低 | 高（Matern 更灵活） |
| 残差建模 | 全局单一 GPR | 全局 GPR + 分层 GPR 微调 |
| 长度尺度 | 全局统一 | 各层独立（但从全局初始化） |

**核心差异**：CSP-RK-HLG 引入 Matern 核函数增强空间自相关建模能力，并通过分层 GPR 微调策略避免 HGP-RK 的小样本问题，同时捕捉不同浓度层残差的空间自相关差异。

## 预期效果
- R² >= 0.856（比 CSP-RK 的 0.8535 提升约 0.0025）
- MAE <= 6.9, RMSE <= 10.8
- 特别在不同浓度层空间自相关特性差异明显的数据

## 方法指纹
MD5: `csp_rk_hybrid_layer_gpr_v1`

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
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel

def csp_rk_hlg_fit(m_train, y_train, X_train, T1, T2, nu=2.5, fine_tune_ratio=0.3):
    """CSP-RK-HLG 训练"""
    poly = PolynomialFeatures(degree=2, include_bias=False)
    layers = np.where(m_train < T1, 0, np.where(m_train < T2, 1, 2))

    layer_models = {}
    all_residuals = []

    # Step 1: 分层 OLS
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

        layer_models[layer] = {
            'ols': ols,
            'X': X_layer,
            'residual': residual_layer
        }
        all_residuals.append(residual_layer)

    residual_all = np.concatenate(all_residuals)

    # Step 2: 全局 GPR 初始化
    kernel_global = ConstantKernel(10.0) * RBF(length_scale=15.0) + WhiteKernel(noise_level=1.0)
    gpr_global = GaussianProcessRegressor(kernel_global, n_restarts_optimizer=2, alpha=0.1)
    gpr_global.fit(X_train, residual_all)

    # 提取全局长度尺度
    length_scale_global = gpr_global.kernel_.theta[1] if hasattr(gpr_global.kernel_, 'theta') else 15.0

    # Step 3: 分层 Matern GPR 微调
    gpr_layers = {}
    for layer, model_info in layer_models.items():
        X_layer = model_info['X']
        residual_layer = model_info['residual']

        # 使用 Matern 核，从全局初始化长度尺度
        kernel_layer = ConstantKernel(10.0) * Matern(length_scale=length_scale_global, nu=nu) + WhiteKernel(noise_level=1.0)
        gpr_layer = GaussianProcessRegressor(kernel_layer, n_restarts_optimizer=1, alpha=0.5)
        gpr_layer.fit(X_train, residual_layer)  # 用全量 X，但只建模当前层残差
        gpr_layers[layer] = gpr_layer

    return layer_models, gpr_global, gpr_layers, poly

def csp_rk_hlg_predict(m_test, X_test, layer_models, gpr_global, gpr_layers, poly, T1, T2, fine_tune_ratio=0.3):
    """CSP-RK-HLG 预测"""
    layers = np.where(m_test < T1, 0, np.where(m_test < T2, 1, 2))
    n = len(m_test)

    # OLS 预测
    m_poly = poly.transform(m_test.reshape(-1, 1))
    M_cal = np.zeros(n)
    for layer_id, model_info in layer_models.items():
        mask = layers == layer_id
        if np.sum(mask) > 0:
            M_cal[mask] = model_info['ols'].predict(m_poly[mask])

    # GPR 混合预测
    R_global = gpr_global.predict(X_test)

    R_layer_sum = np.zeros(n)
    for layer_id, gpr_layer in gpr_layers.items():
        R_layer_pred, _ = gpr_layer.predict(X_test, return_std=True)
        mask = layers == layer_id
        R_layer_sum[mask] = R_layer_pred[mask]

    # 混合
    R_pred = R_global + fine_tune_ratio * (R_layer_sum - R_global)

    return M_cal + R_pred
```

## 交叉验证策略
十折交叉验证时，使用固定的 Matern nu=2.5 和 fine_tune_ratio=0.3 参数。混合 GPR 策略比纯分层 GPR 更稳定，不容易在小样本上过拟合。
