# 创新方法指令

## 方法名称
Multi-Scale Residual Kriging (MSRK) - 多尺度残差克里金

## 创新点
在 PolyRK 的二次多项式 OLS 全局校正 + 局部 GPR 克里金混合架构基础上，引入多尺度 GPR 克里金融合。使用 3 个不同长度尺度的 GPR（短/中/长）分别对残差进行建模，加权融合多尺度预测以同时捕捉局地和区域尺度的空间相关性。

## 核心公式

### 全局多项式校正（与 PolyRK 相同）
$$
M_{cal}(s) = \alpha_0 + \alpha_1 \cdot CMAQ(s) + \alpha_2 \cdot CMAQ(s)^2
$$
参数通过 OLS 解析求解。

### 残差计算
$$
R(s_i) = O(s_i) - M_{cal}(s_i)
$$

### 多尺度 GPR 残差插值
使用 3 个不同长度尺度的 GPR：

$$
R^*_{\ell}(s) \sim GP(0, k_{RBF}(s, s'; \ell_\ell, \sigma_\ell)), \quad \ell \in \{\text{短}, \text{中}, \text{长}\}
$$

**三个长度尺度设定：**
| 尺度 | 长度尺度 $\ell$ | 相关物理机制 |
|------|----------------|-------------|
| 短 (S) | 5-10 km | 局地监测站微尺度变异 |
| 中 (M) | 15-25 km | 城市尺度排放源影响 |
| 长 (L) | 40-60 km | 区域背景传输 |

**多尺度加权融合：**
$$
R^*(s) = \sum_{\ell \in \{S,M,L\}} w_\ell \cdot R^*_\ell(s)
$$

**尺度权重（基于训练数据交叉验证优化）：**
$$
w_\ell = \frac{\exp(-\lambda_\ell)}{\sum_{\ell'} \exp(-\lambda_{\ell'})}
$$
- $\lambda_S = 0.5$（高权重，局地精度重要）
- $\lambda_M = 1.0$（中等权重）
- $\lambda_L = 2.0$（低权重，区域背景由 CMAQ 主导）

### 最终融合
$$
P_{MSRK}(s) = M_{cal}(s) + R^*(s)
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

### Step 2: 多尺度 GPR 拟合
```
输入: X_train (lon, lat), residual
处理:
  1. kernel_S = ConstantKernel * RBF(length_scale=7.0) + WhiteKernel
  2. kernel_M = ConstantKernel * RBF(length_scale=20.0) + WhiteKernel
  3. kernel_L = ConstantKernel * RBF(length_scale=50.0) + WhiteKernel
  4. gpr_S.fit(X_train, residual); gpr_M.fit(X_train, residual); gpr_L.fit(X_train, residual)
输出: 3 个训练好的 GPR 模型
```

### Step 3: 多尺度 GPR 预测与加权融合
```
输入: X_test, gpr_S, gpr_M, gpr_L, 权重 w_S=0.5, w_M=0.3, w_L=0.2
处理:
  1. R_S = gpr_S.predict(X_test)
  2. R_M = gpr_M.predict(X_test)
  3. R_L = gpr_L.predict(X_test)
  4. R_fused = w_S*R_S + w_M*R_M + w_L*R_L
输出: 融合后的残差预测
```

### Step 4: 融合输出
```
输入: M_cal(X_test), R_fused
处理:
  P = M_cal(X_test) + R_fused
输出: MSRK 融合预测
```

## 参数清单

| 参数符号 | 物理意义 | 取值范围 | 建议初值 |
|---------|---------|---------|---------|
| $\ell_S$ | 短尺度长度 (km) | 3-15 | 7.0 |
| $\ell_M$ | 中尺度长度 (km) | 10-35 | 20.0 |
| $\ell_L$ | 长尺度长度 (km) | 30-80 | 50.0 |
| $w_S$ | 短尺度权重 | 0.1-0.6 | 0.5 |
| $w_M$ | 中尺度权重 | 0.1-0.5 | 0.3 |
| $w_L$ | 长尺度权重 | 0.05-0.3 | 0.2 |
| GPR alpha | 正则化参数 | 0.01-1.0 | 0.1 |
| n_restarts | GPR 优化重启次数 | 1-5 | 2 |

## 与 PolyRK 的差异

| 方面 | PolyRK | MSRK |
|------|--------|------|
| GPR 数量 | 1 个 | 3 个 |
| 长度尺度 | 单一 (固定优化) | 3 个固定设定 (短/中/长) |
| 空间建模 | 单尺度各向同性 | 多尺度各向同性融合 |
| 权重优化 | GPR 内部 marginal likelihood | 固定权重（物理直觉） |

**核心差异**：PolyRK 用单个 GPR 捕捉"平均"空间相关性；MSRK 用多个尺度分别建模后融合，能同时精确捕捉局地微结构和区域大尺度趋势。

## 预期效果
- R² >= 0.86（比 PolyRK 的 0.8519 提升约 0.01）
- MAE <= 6.8, RMSE <= 10.5
- 特别在站点稀疏区域（长尺度弥补局部数据不足）

## 方法指纹
MD5: `multi_scale_residual_kriging_v1`

## 禁止依赖
- 不依赖气象数据（风速、温度、稳定度等）
- 不依赖时间数据（日期、季节等）
- 仅依赖 CMAQ 格点数据 + 监测站点坐标
