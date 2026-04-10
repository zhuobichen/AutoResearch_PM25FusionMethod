# 时空残差共克里金 (Spatio-Temporal Residual Co-Kriging, STRK)

## 1. 方法名称

- **中文**: 时空残差共克里金
- **英文**: Spatio-Temporal Residual Co-Kriging (STRK)
- **版本**: v1.0
- **发布日期**: 2026-04-09

## 2. 核心公式

### 2.1 时空变异函数

$$\Gamma(\mathbf{h}_s, h_t; \boldsymbol{\rho}, \tau) = \rho_s \cdot \exp\left(-\frac{|\mathbf{h}_s|}{\lambda_s}\right) \cdot \exp\left(-\frac{h_t}{\tau}\right)$$

- $\mathbf{h}_s$: 空间距离向量
- $h_t$: 时间间隔
- $\lambda_s$: 空间相关长度
- $\tau$: 时间相关尺度
- $\rho_s$: 空间方差贡献

### 2.2 残差时空插值

$$R^*(\mathbf{x}, t) = \sum_{i=1}^{N} \sum_{j=1}^{T} w_{ij} \cdot R(\mathbf{x}_i, t_j)$$

其中权重 $w_{ij}$ 由时空变异函数决定:

$$w_{ij} = \frac{\Gamma(\mathbf{x} - \mathbf{x}_i, t - t_j)}{\sum_{k,l} \Gamma(\mathbf{x}_k - \mathbf{x}_l, t_k - t_l)}$$

### 2.3 最终融合预测

$$Z^*(\mathbf{x}, t) = Z_{RK}(\mathbf{x}, t) + \sum_{k=1}^{K} \theta_k \cdot R_k^*(\mathbf{x}, t)$$

其中 $Z_{RK}$ 为 RK-Poly 基础预测，$R_k^*$ 为第 $k$ 个残差分量的时空插值。

### 2.4 残差分解

$$R(\mathbf{x}, t) = R_{systematic}(\mathbf{x}) + R_{temporal}(t) + R_{spatio-temporal}(\mathbf{x}, t)$$

- $R_{systematic}$: 空间系统性偏差 (如排放源影响)
- $R_{temporal}$: 时间周期性偏差 (如日变化)
- $R_{spatio-temporal}$: 时空交互偏差

## 3. 参数清单

| 参数符号 | 物理意义 | 取值范围 | 建议初值 |
|---------|---------|---------|---------|
| $\lambda_s$ | 空间相关长度 (km) | 10-50 | 20.0 |
| $\tau$ | 时间相关尺度 (h) | 1-12 | 3.0 |
| $\rho_s$ | 空间方差权重 | 0.3-0.8 | 0.5 |
| $\theta_1$ | 系统性残差权重 | 0.1-0.5 | 0.3 |
| $\theta_2$ | 时间残差权重 | 0.1-0.3 | 0.15 |
| $\theta_3$ | 时空交互权重 | 0.1-0.4 | 0.25 |
| $\rho_{nugget}$ | 块金效应 | 0.01-0.2 | 0.08 |
| $\rho_{sill}$ | 基台值 | 0.5-2.0 | 1.0 |
| $T_{window}$ | 时间窗口 (h) | 6-24 | 12 |

## 4. 关键步骤

### Step 1: 残差提取

```
输入: RK-Poly 预测值 Z_RK, 监测值 Z_obs
处理:
  1. R(x,t) = Z_obs(x,t) - Z_RK(x,t)
  2. 分解残差:
     - R_systematic(x) = mean_t[R(x,t)]
     - R_temporal(t) = mean_x[R(x,t)] - mean[R]
     - R_st(x,t) = R(x,t) - R_systematic(x) - R_temporal(t)
输出: 三类残差分量
```

### Step 2: 时空变异函数建模

```
输入: 残差分量, 时空坐标
处理:
  1. 计算经验变异函数:
     γ(h_s, h_t) = (1/2N) × Σ [R(x_i,t_i) - R(x_j,t_j)]²
  2. 拟合理论模型:
     Γ(h_s,h_t;λ_s,τ,ρ) = ρ × exp(-h_s/λ_s) × exp(-h_t/τ)
  3. 参数估计: λ_s, τ, ρ 通过加权最小二乘确定
输出: 时空变异函数参数
```

### Step 3: 残差时空插值

```
输入: 残差分量, 变异函数参数, 目标时空点
处理:
  1. 构建时空邻域 (空间k近邻, 时间窗口内)
  2. 解克里金方程组:
     [Γ] × w = γ_target
  3. 计算插值残差:
     R^*(x,t) = Σ w_ij × R(x_i,t_j)
输出: 各残差分量插值
```

### Step 4: 融合预测

```
输入: Z_RK(x,t), 残差插值, 权重参数
处理:
  1. R_final^*(x,t) = θ_1×R_sys^*(x) + θ_2×R_temp^*(t) + θ_3×R_st^*(x,t)
  2. Z_final(x,t) = Z_RK(x,t) + R_final^*(x,t)
输出: STRK 最终预测
```

## 5. 物理意义

### 5.1 残差时空相关性

- **空间相关**: CMAQ 偏差在相邻区域相似 (排放源区域偏差持续)
- **时间相关**: 日间/夜间偏差模式重复出现 (日变化规律)
- **时空交互**: 早高峰排放 + 稳定大气 = 高偏差累积

### 5.2 物理约束

- **质量守恒**: 残差校正不改变总体均值
- **传输物理**: 偏差随主导风向传递
- **湍流混合**: 稳定度差时空间均匀化慢

## 6. 方法指纹 (MD5)

```
fd405bb1f1ebe15da25c5590640ecc6a
```

指纹生成公式:
```
MD5("Spatio-Temporal Residual Co-Kriging|
     Z*(x,t) = Z_RK(x,t) + Σθk·Γ(x-xk,t-tk;ρ,τ)|
     ρ_spatial, τ_temporal, θ_residual, nugget, sill")
```

## 7. 预期性能

| 指标 | 目标值 | 基准 (RK-Poly) |
|------|--------|---------------|
| R² | ≥ 0.8619 | 0.8519 |
| MAE | ≤ 7.0 | 7.09 |
| RMSE | ≤ 10.9 | 11.05 |

## 8. 创新点总结

1. **残差时空建模**: 捕捉 CMAQ 系统性偏差的时空传播规律
2. **多分量分解**: 分离空间/时间/时空交互偏差
3. **共克里金框架**: 利用残差空间相关性增强插值精度
4. **无权重学习**: 权重基于残差物理特性设定
