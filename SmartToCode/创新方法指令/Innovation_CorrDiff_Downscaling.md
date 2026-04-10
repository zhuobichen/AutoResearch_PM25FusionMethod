# 创新方法指令

## 方法名称
CorrDiff-3km：残差修正扩散降尺度法 (Residual Corrective Diffusion for 3km Downscaling)

## 方法来源
论文：China Regional 3km Downscaling Based on Residual Corrective Diffusion Model (Sun et al., 2025)
arXiv: 2512.05377v4

## 创新核心
使用扩散概率模型学习从25km CMAQ/GFS到3km高分辨率的残差校正场，替代传统的确定性降尺度方法。生成式模型能够捕捉降尺度过程中的不确定性，输出多个可能的降尺度结果。

## 输入数据
- 粗分辨率CMAQ：shape (lat_coarse, lon_coarse, time) - 25km
- 气象场GFS：shape (lat_coarse, lon_coarse, time, n_met) - 25km
- 高分辨率CMAQ-MESO：shape (lat_fine, lon_fine, time) - 3km（训练用）
- 监测站数据：shape (n, time)

## 输出数据
- 降尺度结果：shape (lat_fine, lon_fine, time) - 3km
- 不确定性估计：多个采样路径的统计量

## 核心公式

### 扩散前向过程（加噪）
$$
q(x_t | x_{t-1}) = Normal(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$
其中 $\beta_t$ 是噪声调度（cosine schedule）

### 反向过程（去噪网络）
$$
p_\theta(x_{t-1} | x_t, c) = Normal(\mu_\theta(x_t, t, c), \sigma_t^2 I)
$$
其中 $c$ 是条件信息（粗分辨率CMAQ + 气象场）

### 条件引导机制（无分类器引导）
$$
\epsilon_\theta(x_t, t, c) = (1+w) \epsilon_\theta(x_t, t, c) - w \epsilon_\theta(x_t, t, \emptyset)
$$
$w$ 是引导强度，$\emptyset$ 是空条件

### 残差预测目标
$$
\epsilon^* = \epsilon_\theta(x_t, t, c) - \epsilon_{CMAQ}(x_t^{coarse}, t)
$$
实际学习的是CMAQ无法提供的高频残差

## 关键步骤

### Step 1: 配对数据构建
```
输入: CMAQ_coarse (25km), CMAQ_MESO (3km)
处理:
  1. 将 CMAQ_coarse 双线性插值到 3km 网格
  2. 计算残差: R = CMAQ_MESO - CMAQ_coarse_interp
  3. 提取高频成分: R_high = R - gaussian_blur(R, sigma=5km)
  4. 构建条件图: c = [CMAQ_coarse_interp, wind, T, humidity]
输出: (R_high, c) 训练对
```

### Step 2: 扩散模型训练
```
输入: R_high, c, time_steps=1000
处理:
  1. 初始化 UNet 骨架网络（3个下/上采样块）
  2. 条件注入: 在每个块注入 c 的 embedding
  3. 前向过程: 对每个 batch 随机采样 t, 加噪 x_t
  4. 损失: L = ||epsilon - epsilon_theta(x_t, t, c)||^2
  5. 训练 100 epochs, batch_size=16
输出: 训练好的 diffusion model
```

### Step 3: 推理降尺度
```
输入: CMAQ_coarse, GFS_met, n_samples=10
处理:
  1. 插值 CMAQ_coarse 到 3km
  2. 初始化 x_T ~ N(0, I)
  3. for t = T to 1:
     - epsilon = model(x_t, t, c)
     - if t > 0: x_{t-1} = (x_t - sqrt(1-alpha_t)*epsilon) / sqrt(alpha_t) + sigma_t * noise
     - else: x_0 = pred_residual
  4. 重复 n_samples 次
输出: n_samples 个降尺度结果
```

### Step 4: 融合输出
```
输入: CMAQ_interp, residual_samples, station_obs
处理:
  1. 计算残差统计: mean_R, std_R
  2. 可选：用监测站数据校正 mean_R
  3. 最终降尺度: P = CMAQ_interp + mean_R
  4. 不确定性: P_upper = P + 1.96*std_R, P_lower = P - 1.96*std_R
输出: P, P_upper, P_lower
```

## 【创新点】

1. **无权重学习**：扩散模型学习残差分布，非参数化权重
2. **生成式不确定性**：天然输出多个可能的降尺度结果
3. **高频保留**：传统插值会平滑高频细节，扩散模型能学习高频模式
4. **物理一致性**：条件注入保持与粗分辨率的一致性

## 创新判定
- 是否使用权重学习（Ridge/Lasso/线性回归）？否
- 是否有物理可解释性？是（残差=真值-模型预测，有物理含义）
- 是否保留：创新成立

## 创新优势
- 相比简单插值（CMAQ双线性）：能学习非线性高频校正
- 相比确定性降尺度：不确定性量化更合理
- 预期R²提升 >= 0.02（在3km网格上）

## 风险假设
- 训练需要配对的25km/3km数据
- 推理计算成本高（1000步迭代）
- 验证计划：对比双线性插值的空间功率谱

## 方法指纹
MD5: `corrdiff_residual_diffusion_v1`

## 参数清单

| 参数符号 | 物理意义 | 取值范围 | 建议初值 |
|---------|---------|---------|---------|
| T | 扩散步数 | 500-2000 | 1000 |
| w | 引导强度 | 0.0-2.0 | 1.0 |
| n_samples | 采样数量 | 1-20 | 10 |
| lr | 学习率 | 1e-5-1e-4 | 1e-4 |
| batch_size | 批大小 | 8-32 | 16 |
| epochs | 训练轮数 | 50-200 | 100 |

## 输入输出格式
- 输入：CMAQ coarse netCDF + GFS气象 netCDF + CMAQ-MESO fine netCDF（训练）
- 输出：3km降尺度PM2.5 + 不确定性区间
- 支持多采样路径统计

## 工具依赖
- PyTorch (CUDA)
- diffusers library
- xarray for netCDF 处理
