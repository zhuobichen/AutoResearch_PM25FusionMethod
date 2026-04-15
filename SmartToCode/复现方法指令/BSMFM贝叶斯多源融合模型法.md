# 方法名称
贝叶斯多源融合模型 (Bayesian Multisource Fusion Model, BSMFM)

## 类型
复现

## 核心公式

### 1. 贝叶斯层次模型结构
$$
Y_i(s,t) = \underbrace{X_i(s,t)}_{\text{潜在场}} + \underbrace{\epsilon_i(s,t)}_{\text{测量误差}}
$$

其中 $Y_i$ 是第 $i$ 个数据源的观测值，$X_i$ 是对应的潜在真值。

### 2. 潜在场建模
$$
X_i(s,t) = \underbrace{f_i(s,t)}_{\text{确定性成分}} + \underbrace{\xi(s,t)}_{\text{共享随机场}} + \underbrace{\eta_i(s,t)}_{\text{源特异性偏差}}
$$

### 3. 共享时空随机场
$$
\xi(s,t) \sim SP(\Sigma, \Theta)
$$
使用时空协方差函数建模，考虑空间相关性（指数协方差）和时间相关性（AR1）。

### 4. CMAQ模型偏差校正
$$
f_{CMAQ}(s,t) = \beta_0 + \beta_1 \cdot CMAQ(s,t) + \xi_{CMAQ}(s,t)
$$
其中 $\xi_{CMAQ}$ 是CMAQ的空间偏移场。

### 5. 后验推断
使用变分推断或MCMC进行后验采样：
$$
p(\Theta | Y) \propto p(Y | \Theta) p(\Theta)
$$

## 算法步骤

1. **步骤1：数据预处理**
   - 将所有数据源统一到1km网格
   - 处理空间不对齐（upscaling方法）
   - 质量控制与异常值剔除

2. **步骤2：构建层次模型**
   - 定义潜在时空场 $\xi(s,t)$
   - 为每个CMAQ模型建立偏差校正方程
   - 指定测量误差分布

3. **步骤3：参数估计**
   - 使用INLA或MCMC进行贝叶斯推断
   - 学习空间相关性参数
   - 估计源特异性偏差

4. **步骤4：融合预测**
   - 合成所有数据源信息
   - 生成1km分辨率PM2.5地图
   - 输出不确定性估计

## 参数说明
- spatial_range：空间相关尺度(km)，由变异函数估计
- temporal_range：时间相关尺度(day)，由自相关分析估计
- nugget_variance：微尺度变异，由残差估计
- beta_0, beta_1：线性偏差校正系数，由贝叶斯推断估计
- data_sources：融合的数据源数量，默认2+
- resolution：输出空间分辨率，默认1km

## 预期效果
- R2：0.75-0.82
- RMSE：12-18 μg/m³
- MB：±2 μg/m³

## 验证方案
十折交叉验证：按时间划分十折，轮流使用9折训练、1折验证，评估时间外推能力。
