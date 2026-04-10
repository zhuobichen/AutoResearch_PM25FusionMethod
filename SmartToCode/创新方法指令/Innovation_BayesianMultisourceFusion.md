# 创新方法指令

## 方法名称
BMSF-Geostat：贝叶斯多源融合地球统计法 (Bayesian Multisource Fusion with Geostatistical Mapping)

## 方法来源
论文：A Bayesian Multisource Fusion Model for Spatiotemporal PM2.5 in an Urban Setting (Riley et al., 2025)
arXiv: 2506.10688v2

## 创新核心
使用贝叶斯层级模型融合多个PM2.5数据源（大气模型输出、卫星AOD、植被指数），通过潜在时空随机场建模空间错位问题，使用上尺度方法实现跨分辨率预测。

## 输入数据
- 监测站坐标：shape (n, 2)
- CMAQ数据：shape (lat, lon, time) - 1个大气模型
- CTM2数据：shape (lat, lon, time) - 第2个大气模型
- AOD数据：shape (lat_aod, lon_aod, time) - 卫星AOD
- 植被指数NDVI：shape (lat, lon) - 静态协变量
- 监测PM2.5：shape (n, time)

## 输出数据
- 融合结果：shape (lat_fine, lon_fine, time) - 1km高分辨率

## 核心公式

### 贝叶斯层级模型

**数据层（似然）：**
$$
Y_i(t) \sim Normal(\mu(s_i, t), \sigma_y^2)
$$
$$
\mu(s, t) = f(s, t) + \beta_1 \cdot CMAQ(s, t) + \beta_2 \cdot CTM2(s, t) + \beta_3 \cdot AOD(s, t)
$$

**过程层（潜在场）：**
$$
f(s, t) = \phi * f(s, t-1) + \eta(s, t)
$$
$$
\eta(s, t) \sim SPDE(Gaussian\ Process)
$$

**参数层（先验）：**
$$
\beta_k \sim Normal(0, \sigma_\beta^2), \quad k=1,2,3
$$
$$
\sigma_y^2 \sim InverseGamma(a, b)
$$

### 空间错位处理（上尺度方法）
对于不同分辨率的预测目标：
$$
\mu_{fine}(s, t) = B(s) \cdot \mu_{coarse}(s, t) + \epsilon_{scale}(s)
$$
其中 $B(s)$ 是空间变化的偏差校正矩阵

### INLA求解（无需MCMC）
使用 Integrated Nested Laplace Approximation 近似后验：
$$
\pi(\theta | y) \approx \int \pi(\theta, f | y) df
$$

## 关键步骤

### Step 1: 数据预处理与对齐
```
输入: CMAQ, CTM2, AOD, NDVI, stations
处理:
  1. 将所有数据重网格化到统一1km分辨率
  2. 对AOD进行插值填补（仅白天有数据）
  3. 提取每个监测站位置的多源协变量值
  4. 标准化处理: z = (x - mean) / std
输出: 对齐后的数据集
```

### Step 2: SPDE近似构建潜在场
```
输入: station_coords, time_points
处理:
  1. 定义 Matérn 协方差函数
  2. 使用 SPDE 方法离散化随机偏微分方程
  3. 构建精度矩阵 Q (sparse)
  4. 时间演化: f(t) = A * f(t-1) + noise
输出: SPDE 图结构
```

### Step 3: 贝叶斯层级模型拟合
```
输入: Y_observed, CMAQ, CTM2, AOD, SPDE_struc
处理:
  1. 定义缺陷坐标: mu = X*beta + f_spde
  2. 设置弱信息先验: beta ~ N(0, 10), sigma ~ IG(1, 0.001)
  3. 使用 INLA 近似后验分布
  4. 提取后验均值和 credible interval
输出: beta 后验, f_spde 后验
```

### Step 4: 网格预测
```
输入: beta_posterior, f_spde_posterior, new_locations
处理:
  1. 在新位置计算 SPDE 预测
  2. 融合协变量: mu_new = X_new * beta_mean + f_new
  3. 计算不确定性: var_new = X_new * Sigma_beta * X_new' + var_f
输出: mu_new, lower, upper (95% CI)
```

## 【创新点】

1. **无权重学习**：beta系数有概率解释（后验均值+方差），非点估计权重
2. **潜在时空随机场**：通过SPDE建模，不用网格化GP，更高效
3. **不确定性量化**：完整后验分布，不仅仅是点预测
4. **多源融合**：同时利用多个大气模型和卫星数据

## 创新判定
- 是否使用权重学习（Ridge/Lasso/线性回归）？否（贝叶斯后验）
- 是否有物理可解释性？是（潜在场+协变量系数）
- 是否保留：创新成立

## 创新优势
- 相比StackingEnsemble：贝叶斯框架提供不确定性，而非点估计权重
- 相比GenFriberg：SPDE潜在场比GVN更灵活，支持时空演化
- 预期R²提升 >= 0.015

## 风险假设
- INLA近似对高度非线性问题可能不准确
- 计算成本较高（全后验需积分）
- 验证计划：对比贝叶斯后验均值与普通OLS的预测区间覆盖

## 方法指纹
MD5: `bayesian_multisource_fusion_spde_v1`

## 参数清单

| 参数符号 | 物理意义 | 取值范围 | 建议初值 |
|---------|---------|---------|---------|
| range | SPDE 相关距离 (km) | 20-200 | 80.0 |
| sigma | SPDE 方差 | 1-50 | 15.0 |
| phi | 时间自相关 | 0.5-0.99 | 0.8 |
| alpha | Matérn 参数 | 1-2 | 1.5 |
| beta1 | CMAQ 系数 | 0-2 | 1.0 |
| beta2 | CTM2 系数 | 0-2 | 0.0 |

## 输入输出格式
- 输入：监测数据CSV + CMAQ netCDF + CTM2 netCDF + AOD netCDF + NDVI tiff
- 输出：融合网格PM2.5 + 95%可信区间
- 支持贝叶斯后验采样

## 工具依赖
- INLA R包（可通过rpy2调用）
- PySPDE（可选，Python替代）
