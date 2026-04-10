# 创新方法指令

## 方法名称
PolyGPR-Adapt：大气稳定度自适应多项式-高斯过程残差融合法
(Atmospheric-Stability-Adaptive Polynomial Calibration with Gaussian Process Residual Modeling)

## 输入数据
- 监测站坐标：shape (n, 2)
- CMAQ数据：shape (lat, lon, time)
- 气象数据（温度、边界层高度、风速、湿度）：shape (lat, lon, time, n_met)
- 稳定度等级（Pasquill-Gifford分类）：shape (lat, lon, time)

## 输出数据
- 融合结果：shape (lat, lon, time)

## 核心公式

### 第一步：多项式CMAQ校正
$$
CMAQ_{cal}(s) = \alpha_0 + \alpha_1 \cdot CMAQ(s) + \alpha_2 \cdot CMAQ(s)^2 + \alpha_3 \cdot T(s) + \alpha_4 \cdot PBLH(s)
$$
其中T=温度，PBLH=边界层高度，参数通过监测站数据拟合（确定性，最小二乘解析解）

### 第二步：残差计算
$$
R(s_i) = O(s_i) - CMAQ_{cal}(s_i), \quad \forall \text{ station } i
$$

### 第三步：大气稳定度自适应GPR
$$
R(s) \sim GP(0, k_{Mattern}(s, s'; \theta_{stab}))
$$

**稳定度自适应的变异函数参数：**
$$
\gamma(h; \sigma, \ell, stab) = \sigma^2 \left[1 - \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{h \cdot \phi(stab)}{\ell}\right)^\nu K_\nu\left(\frac{h \cdot \phi(stab)}{\ell}\right)\right]
$$
其中：
- $\ell$ = 基线相关长度
- $\phi(stab)$ = 稳定度缩放因子（A/B类不稳定→大值，C/D类中性→标准，E/F类稳定→小值）
- $\nu$ = Mattern阶数（默认ν=3/2）
- $K_\nu$ = 第二类修正Bessel函数

**稳定度缩放因子表：**
| 稳定度 | Pasquill类 | φ(stab) |
|--------|-----------|---------|
| 极不稳定 | A | 2.5 |
| 不稳定 | B | 1.8 |
| 弱不稳定 | C | 1.3 |
| 中性 | D | 1.0 |
| 弱稳定 | E | 0.7 |
| 稳定 | F | 0.4 |

### 第四步：融合结果
$$
P_{fusion}(s) = CMAQ_{cal}(s) + \hat{R}(s)
$$
其中 $\hat{R}(s)$ 由GPR后验均值给出

### 不确定性估计
$$
\sigma^2_{fusion}(s) = \sigma^2_{CMAQcal} + \sigma^2_{GPR}(s)
$$
GPR后验方差直接提供空间不确定估计

## 关键步骤
1. **多项式拟合**：在监测站上用最小二乘拟合α参数（解析解，无迭代）
2. **残差提取**：计算每个站点的校正后残差
3. **稳定度分类**：基于气象数据确定每个网格点的Pasquill稳定度等级
4. **GPR拟合**：用监测站残差拟合GPR，变异函数参数根据稳定度自适应调整
5. **网格预测**：GPR后验均值+方差输出到网格
6. **十折验证**：对比不同稳定度设定下的CV-RMSE

## 【创新点】
1. **无权重学习**：所有参数通过解析最小二乘（GPR超参数优化通过marginal likelihood）确定
2. **大气稳定度物理解释**：稳定度直接影响大气扩散率，决定空间相关长度
3. **不确定性量化**：GPR天然提供后验方差，可用于不确定性传播分析
4. **相比普通ResidualKriging**：变异函数参数固定→自适应，稳定度分类提供物理依据

## 创新判定
- 是否使用权重学习（Ridge/Lasso/线性回归）？否（多项式为解析最小二乘，GPR为边缘似然优化）
- 是否有物理可解释性？是（大气稳定度直接影响扩散率/相关长度）
- 是否保留：创新成立

## 创新优势
- 相比FCopt（优化融合）：无需学习时变权重，物理参数稳定
- 相比GenFriberg（多步融合）：变异函数随大气状态自适应，非固定参数
- 预期R²提升 >= 0.02（稳定度自适应GPR比普通克里金更好地捕捉非稳态条件）

## 风险假设
- 稳定度分类的准确性影响GPR参数
- 对于极端污染事件（稳定强天），残差可能超出GPR假设
- 验证计划：对比固定参数GPR（Innovation_ResidualKriging）的十折CV-RMSE

## 方法指纹
MD5: `polygpr_adapt_atmospheric_stability_v1`

## 输入输出格式
- 输入：监测数据CSV + CMAQ netCDF + 气象netCDF
- 输出：融合网格PM2.5 + 不确定性网格σ
- 支持十折交叉验证
