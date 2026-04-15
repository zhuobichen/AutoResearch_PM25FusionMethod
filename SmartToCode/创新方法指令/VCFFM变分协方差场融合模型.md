# 方法名称
变分协方差场融合模型 (Variational Covariance Field Fusion Model, VCFFM)

## 类型
创新

## 核心公式

### 1. 多源潜在表示学习
$$
p(z | Y_{CMAQ}, Y_{obs}) = \mathcal{N}(\mu, \sigma^2)
$$
变分自编码器学习CMAQ输出 $Y_{CMAQ}$ 与监测数据 $Y_{obs}$ 的共享潜在空间表示 $z$。

### 2. 克里金协方差约束项
在变分推断中融入空间协方差结构：
$$
\log p(K_{spatial} | z) = -\frac{1}{2} z^T \Sigma^{-1} z - \frac{1}{2} \log |\Sigma|
$$
$\Sigma$ 为基于变异函数的协方差矩阵，保证生成场空间一致性。

### 3. 融合解码器重建
$$
\hat{Y}_{fusion}(s) = D_\theta(z(s)) = \beta_0 + \beta_1 \cdot Y_{CMAQ}(s) + \beta_2 \cdot G_\phi(z(s))
$$
解码器 $D_\theta$ 融合CMAQ物理场 $Y_{CMAQ}$ 与潜在表示 $z$ 的非线性变换 $G_\phi$。

### 4. 多尺度分解损失
$$
\mathcal{L}_{multi} = \sum_{l=1}^{L} \lambda_l \|\hat{Y}_{fusion}^{(l)} - Y_{obs}^{(l)}\|^2
$$
分尺度 $l$ 计算损失，捕捉不同空间变异尺度（宏观趋势、中尺度变化、微尺度波动）。

### 5. 不确定性量化
$$
\sigma_{fusion}^2(s) = \sigma_{CMAQ}^2(s) + \sigma_{encoder}^2(z) + \sigma_{spatial}^2(s)
$$
总不确定性由模型不确定性、编码器不确定性和空间插值不确定性组成。

## 算法步骤

1. **步骤1：协方差场构建**
   - 计算监测站间距离矩阵
   - 估计实验变异函数 $\hat{\gamma}(h)$
   - 拟合球状/指数变异函数模型
   - 构建空间协方差矩阵 $\Sigma$

2. **步骤2：变分编码器训练**
   - CMAQ网格特征提取（CNN编码器）
   - 监测站点特征提取（MLP编码器）
   - 变分推断得到潜在分布 $q(z | Y_{CMAQ}, Y_{obs})$
   - KL散度约束保证正则化

3. **步骤3：克里金协方差融合**
   - 在解码器中嵌入协方差约束层
   - 潜在空间采样 $z \sim q(z | \cdot)$
   - 多尺度特征金字塔构建

4. **步骤4：融合重建与不确定性估计**
   - 解码器输出融合网格 $\hat{Y}_{fusion}$
   - Monte Carlo采样估计不确定性
   - 输出PM2.5融合结果及置信区间

## 参数说明
- latent_dim：潜在空间维度，默认32
- encoder_layers：编码器层数，默认3
- decoder_layers：解码器层数，默认3
- covariance_range：协方差变程(km)，由变异函数估计
- covariance_sill：协方差拱高，由变异函数估计
- scale_levels：多尺度分解层数，默认3
- mc_samples：Monte Carlo采样次数，默认50
- beta_kl：KL散度权重，默认0.01
- learning_rate：学习率，默认0.0005

## 预期效果
- R2：0.80-0.86
- RMSE：12-18 μg/m³
- MB：±3 μg/m³
- 不确定性覆盖率：90%以上（实测值落在95%置信区间内）

## 验证方案
十折交叉验证：时空混合划分十折，评估：
1. 融合结果与纯CMAQ的改善程度
2. 不确定性区间的可靠性（区间覆盖率）
3. 不同污染水平下的融合精度
4. 空间尺度保持性（协方差结构是否保留）
