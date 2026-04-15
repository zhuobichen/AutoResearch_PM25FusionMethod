# 方法名称
克里金伪标签增强法 (Kriging-based Pseudo-Label Augmentation)

## 类型
复现

## 核心公式

### 1. 普通克里金插值
$$
\hat{Z}(s_0) = \sum_{i=1}^{n} \lambda_i Z(s_i)
$$
权重 $\lambda_i$ 通过以下约束求解：
$$
\sum_{i=1}^{n} \lambda_i = 1, \quad \sum_{i=1}^{n} \lambda_i C(s_i, s_j) = C(s_0, s_j), \forall j
$$
其中 $C$ 是协方差函数。

### 2. 变异函数建模
$$
\gamma(h) = \begin{cases} 0 & h = 0 \\ c_0 + c \cdot \text{Sph}(h/a) & h > 0 \end{cases}
$$
其中 $c_0$ 是块金值，$c$ 是拱高，$a$ 是变程。

### 3. CNN-RF混合模型
$$
\hat{y}_{CNN-RF} = f_{RF}(f_{CNN}(x; \theta_CNN); \theta_{RF})
$$
CNN提取空间特征，RF进行最终回归。

### 4. 伪标签生成策略
$$
\tilde{Z}(s_{pseudo}) = \hat{Z}_{Krig}(s_{pseudo}), \quad s_{pseudo} \in \text{卫星覆盖区域}
$$
对没有地面监测但有卫星数据的区域生成克里金伪标签。

### 5. 增强训练损失
$$
\mathcal{L}_{aug} = \mathcal{L}_{original} + \alpha \cdot \mathcal{L}_{pseudo}
$$
其中 $\mathcal{L}_{pseudo} = \sum_{s \in S_{pseudo}} |f(x(s)) - \tilde{Z}(s)|$

## 算法步骤

1. **阶段1：空间变异函数估计**
   - 计算实验变异函数 $\hat{\gamma}(h)$
   - 拟合理论模型（球状模型）
   - 验证交叉验证

2. **阶段2：克里金插值**
   - 对观测站点进行克里金插值
   - 生成均匀网格上的插值结果
   - 计算插值标准误差

3. **阶段3：伪标签生成**
   - 识别有AOD覆盖但无监测的区域
   - 使用克里金模型预测PM2.5值
   - 筛选高置信度伪标签（低插值方差）

4. **阶段4：CNN-RF训练**
   - 使用原始标注数据训练CNN特征提取器
   - 使用原始+伪标签数据训练RF回归器
   - 评估泛化性能

## 参数说明
- kriging_range：克里金搜索半径，默认200km
- min_stations：克里金最少站点数，默认10
- nugget：变异函数块金值，默认0.1
- sill：变异函数拱高，默认1.0
- augmentation_ratio：伪标签样本比例，默认0.3-0.5
- CNN_epochs：CNN特征提取训练轮数，默认50
- RF_n_estimators：随机森林树数量，默认100

## 预期效果
- R2：0.72-0.80
- RMSE：18-25 μg/m³
- MB：±5 μg/m³

## 验证方案
十折交叉验证：随机划分十折，评估模型在未见过的监测站上的预测能力。
