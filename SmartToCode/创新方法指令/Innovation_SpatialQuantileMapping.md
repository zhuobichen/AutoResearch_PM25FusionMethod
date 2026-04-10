# 创新方法指令

## 方法名称
Spatial Quantile Deviation Mapping (SQDM) - 空间分位数偏差映射

## 创新点
利用 CMAQ 和监测数据在空间上的联合分布结构，构造空间变动的偏差校正映射。与传统分位数映射（假设偏差全球一致）不同，SQDM 假设偏差是空间的函数 $B(s) = f(CMAQ(s))$，捕捉"同一 CMAQ 预测值在不同位置对应不同偏差"的现象。

## 核心公式

### 空间偏差函数
$$
B(s) = O(s) - CMAQ(s) = \beta_0 + \beta_1 \cdot CMAQ(s) + \beta_2 \cdot Lat(s) + \beta_3 \cdot Lon(s) + \epsilon(s)
$$

### 局地分位数偏差（核心创新）
对于目标点 $s_0$，在邻域 $\{s_i\}$ 内估计：
$$
\hat{B}_{quantile}(s_0, q) = \mathbb{F}_{CMAQ}^{-1}(q) - \mathbb{F}_{O}^{-1}(q)
$$
其中 $\mathbb{F}^{-1}$ 是给定邻域内的分位数函数，$q$ 是 CMAQ 分位数。

### 偏差空间插值
$$
\hat{B}(s_0) = \sum_{i=1}^{n} w_i(s_0) \cdot B(s_i)
$$
权重使用距离反比：
$$
w_i(s_0) = \frac{1/|s_0 - s_i|}{\sum_j 1/|s_0 - s_j|}
$$

### 最终融合
$$
P_{SQDM}(s_0) = CMAQ(s_0) + \hat{B}(s_0) \cdot \phi(CMAQ(s_0))
$$
其中 $\phi$ 是基于 CMAQ 分位数的校正因子：
$$
\phi(CMAQ) = 1 + \gamma \cdot \left(\frac{CMAQ - \overline{CMAQ}}{\sigma_{CMAQ}}\right)
$$

## 关键步骤
1. **邻域选取**：对每个网格点，选取 $k$ 个最近监测站
2. **局地偏差计算**：$B_i = O_i - CMAQ_i$
3. **分位数偏差映射**：
   - 计算邻域内 CMAQ 的分位数
   - 计算对应观测值的分位数
   - 估计 $\hat{B}_{quantile}(q)$
4. **空间插值**：IDW 插值偏差到网格
5. **CMAQ 依赖校正**：根据网格点 CMAQ 值应用分位数校正
6. **融合输出**：$P = CMAQ + \hat{B}$

## 参数清单
- $n_{neighbor}$: 邻域站点数, default: 12
- $\gamma$: CMAQ 依赖校正因子, default: 0.1
- $q_{high}$: 高分位数, default: 0.9
- $q_{low}$: 低分位数, default: 0.1
- $\alpha_{spatial}$: 空间权重指数, default: 2.0

## 预期效果
- SQDM 捕捉"同一 CMAQ 值在不同位置偏差不同"的空间非均匀性
- 预期 R² >= 0.86（相比 RK-Poly 的 0.8519 提升 >= 0.01）
- MAE <= 6.9, RMSE <= 10.8

## 为什么能超越 RK-Poly
RK-Poly 使用 $O = a + bM + cM^2 + \epsilon$ 的全局多项式偏差校正，假设偏差仅与 CMAQ 值相关。SQDM 的创新：
1. 偏差是空间函数 $B(s)$，不是全球常数
2. 分位数映射捕捉 CMAQ-Conc 的非线性关系在空间上的变化
3. 仅用 Lat, Lon, CMAQ, Conc，无气象依赖

## 方法指纹
MD5: `spatial_quantile_deviation_mapping_v1`

## 创新判定
- 是否使用权重学习（Ridge/Lasso/线性回归）？否（仅用 IDW 固定权重）
- 是否有物理可解释性？是（局地偏差反映局地排放源特征）
- 是否依赖气象或时间数据？否
- 创新状态：保留
