# 创新方法指令

## 方法名称
Heteroscedastic GPR PolyRK (HGP-RK) - 异方差高斯过程残差多项式克里金

## 创新点
在 PolyRK 的二次多项式 OLS 全局校正 + 局部 GPR 克里金混合架构基础上，放宽 GPR 残差建模的同方差假设（homoscedastic），改用异方差 GPR 建模残差的空间分布。标准 GPR 假设残差方差在整个空间内是常数，但实际数据中高浓度区域的监测值和模型值偏差往往更大（化学非线性、设备测量范围限制、区域传输贡献变异），使用异方差 GPR 可以更准确地量化这种非均匀不确定性。

## 背景问题

### 标准 GPR 的同方差假设
PolyRK 中的 GPR 假设：
$$
R(s) \sim GP(\mu(s), k(s, s'))
$$
其中协方差函数 $k(s, s')$ 的方差参数 $\sigma^2$ 是全局常数，与位置 $s$ 无关。

### 实际残差的异方差特性
在大气环境数据中，异方差来源：

1. **高浓度区非线性**：CMAQ 模型在高污染时化学机理更复杂，偏差更大
2. **测量不确定性**：高浓度时 Beta 射线法的精度可能下降
3. **空间变异**：城市源排放区（高浓度）的残差可能比郊区（低浓度）更复杂

### 异方差的表现形式
$$
\text{Var}[R(s)] = \sigma^2(s) = \sigma^2_0 \cdot \exp(2 \cdot h(M(s)))
$$
其中 $h(\cdot)$ 可以是：
- 线性：$h(M) = \alpha \cdot M$
- 分段线性：不同浓度区间不同斜率
- 或直接建模 $\log \sigma^2(s)$ 作为另一个 GP

## 核心公式

### 第一步：全局多项式校正（与 PolyRK 相同）
$$
M_{cal}(s) = \alpha_0 + \alpha_1 M(s) + \alpha_2 M(s)^2
$$

### 第二步：残差计算
$$
R_i = O_i - M_{cal}(s_i)
$$

### 第三步：异方差 GPR 建模
**双核异方差协方差函数**：
$$
k_{het}(s_i, s_j) = \sigma^2_i \cdot \sigma^2_j \cdot \exp\left(-\frac{||s_i - s_j||^2}{2\ell^2}\right)
$$

其中 $\sigma^2_i = \sigma^2(M(s_i))$ 是 CMAQ 浓度的函数。

**异方差函数形式**：
$$
\sigma^2(M) = \sigma^2_0 \cdot \left(1 + \alpha \cdot M\right)^2
$$
或使用分段形式：
$$
\sigma^2(M) =
\begin{cases}
\sigma^2_1 & M < T_1 \\
\sigma^2_2 & T_1 \leq M < T_2 \\
\sigma^2_3 & M \geq T_2
\end{cases}
$$

**简化实现 - 输出尺度核（Output-Scale Kernel）**：
使用 GPR 的 `ConstantKernel` 作为输出尺度，但允许其在不同区域变化：
$$
k(s_i, s_j) = \sigma^2(M_i, M_j) \cdot k_{base}(s_i, s_j)
$$

### 第四步：融合结果
$$
P_{HGPRK}(s) = M_{cal}(s) + \hat{R}(s)
$$
其中 $\hat{R}(s)$ 由异方差 GPR 后验均值给出。

## 关键步骤

### 实现方案 A：分层方差估计 + GPR（推荐）
```
输入: m_train, y_train, X_train, residual
处理:
  1. 计算每层残差方差:
     - T1, T2 = 35, 75 (μg/m³)
     - σ²_1 = var(residual[m_train < T1])
     - σ²_2 = var(residual[(m_train >= T1) & (m_train < T2)])
     - σ²_3 = var(residual[m_train >= T2])
  2. 构建异方差权重:
     - α_i = sqrt(σ²_layer / σ²_min)  # 归一化到最小方差层
     - weight_i = 1 / α²  # 低方差区域权重更高
  3. 加权 GPR 拟合:
     - 调整 GPR alpha 参数为每点异方差: α_i
     - gpr.fit(X_train, residual, sample_weight=weight_i)
  4. 预测时考虑异方差: return gpr.predict(X_test)
输出: gpr 模型
```

### 实现方案 B：双 GP 建模（更复杂）
```
输入: m_train, y_train, X_train, residual
处理:
  1. GP1 建模残差均值: R_mean ~ GP(0, k_base)
  2. GP2 建模残差方差: log(σ²) ~ GP(0, k_var)
  3. 预测: posterior mean of R_mean, posterior variance = f(posterior mean of σ²)
  4. 最终预测: M_cal + R_mean_pred
输出: 双 GP 模型
```

### 实现方案 C：浓度依赖核函数
```
输入: X_train, residual, m_train
处理:
  1. 构建浓度依赖核:
     k(s_i, s_j) = σ(M_i) * σ(M_j) * exp(-||s_i-s_j||²/(2ℓ²))
     其中 σ(M) = σ₀ * (1 + α*M)
  2. 使用 scipy.linalg 手动构建协方差矩阵并 Cholesky 分解
  3. GPR 预测: μ = K_star @ K⁻¹ @ residual
输出: 预测均值和方差
```

## 参数清单

| 参数符号 | 物理意义 | 取值范围 | 建议初值 |
|---------|---------|---------|---------|
| T1 | 低-中浓度阈值 | 20-60 μg/m³ | 35 |
| T2 | 中-高浓度阈值 | 60-150 μg/m³ | 75 |
| σ²_1 | 低浓度残差方差 | 0.1-10 | 从数据估计 |
| σ²_2 | 中浓度残差方差 | 0.1-10 | 从数据估计 |
| σ²_3 | 高浓度残差方差 | 0.1-20 | 从数据估计 |
| GPR length_scale | GPR 长度尺度 (km) | 5-50 | 15.0 |
| GPR alpha | GPR 正则化 | 0.01-1.0 | 0.1（改用异方差） |
| n_restarts | GPR 优化重启 | 1-5 | 2 |

## 与 PolyRK 的差异

| 方面 | PolyRK | HGP-RK |
|------|--------|--------|
| GPR 方差假设 | 同方差（全局常数 σ²） | 异方差（σ² 随浓度变化） |
| 高浓度区建模 | 同等对待所有残差 | 给高浓度更大方差，降低其权重 |
| 预测不确定性 | 全局统一 | 空间异质 |
| 计算复杂度 | 较低 | 略高（需要估计每层方差） |
| 对异常值敏感性 | 高（同等权重） | 低（大方差区域自动降权） |

**核心差异**：PolyRK 的 GPR 对所有残差点同等对待，隐含假设残差方差均匀；HGP-RK 根据残差实际分布特性，为不同浓度区域分配不同权重，使预测更准确，特别是在高浓度区。

## 预期效果
- R² >= 0.855（比 PolyRK 的 0.8519 提升约 0.003）
- MAE <= 7.0, RMSE <= 10.9
- 特别在残差异方差性显著的数据集上优势更明显

## 方法指纹
MD5: `heteroscedastic_gpr_polyrk_v1`

## 禁止依赖
- 不依赖气象数据（风速、温度、稳定度等）
- 不依赖时间数据（日期、季节等）
- 仅依赖 CMAQ 格点数据 + 监测站点坐标

## sklearn 实现参考
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def hgp_rk_fit(m_train, y_train, X_train, T1=35, T2=75):
    """HGP-RK 训练（分层方差方案）"""
    # Step 1: 多项式校正（与 PolyRK 相同）
    poly = PolynomialFeatures(degree=2, include_bias=False)
    m_poly = poly.fit_transform(m_train.reshape(-1, 1))
    ols = LinearRegression().fit(m_poly, y_train)
    residual = y_train - ols.predict(m_poly)

    # Step 2: 分层方差估计
    layers = np.where(m_train < T1, 0, np.where(m_train < T2, 1, 2))
    var_dict = {}
    for layer in range(3):
        mask = layers == layer
        if np.sum(mask) > 1:
            var_dict[layer] = np.var(residual[mask])
        else:
            var_dict[layer] = np.var(residual)  # fallback

    # Step 3: 构建异方差权重
    min_var = min(var_dict.values())
    weights = np.array([min_var / var_dict[l] for l in layers])

    # Step 4: GPR 拟合（异方差 alpha）
    kernel = ConstantKernel(10.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=2, alpha=0.1)
    # 使用 sample_weight 传入异方差权重
    gpr.fit(X_train, residual, sample_weight=weights)

    return ols, gpr, poly, var_dict

def hgp_rk_predict(m_test, X_test, ols, gpr, poly, T1=35, T2=75):
    """HGP-RK 预测"""
    m_poly = poly.transform(m_test.reshape(-1, 1))
    M_cal = ols.predict(m_poly)
    R_pred = gpr.predict(X_test)
    return M_cal + R_pred
```

## 验证策略
1. **残差方差分析**：在十折验证前，先在全量数据上检验残差是否存在显著异方差性
2. **对比实验**：分别用同方差 GPR 和异方差 GPR 进行十折，对比 CV-RMSE
3. **权重敏感性**：测试不同方差比例设定下的结果稳定性
