# 创新方法指令

## 方法名称
Concentration-Regime Adaptive Bias Correction (CR-ABC) - 浓度分层自适应偏差校正

## 创新核心
PM2.5偏差呈现非线性特征：模型在低浓度时倾向于高估，高浓度时倾向于低估。CR-ABC通过检测浓度分层，对每个层级独立建模偏差关系，比线性偏差校正（aVNA）更精确捕捉非线性偏差结构。

## 核心公式

### 步骤1：浓度分层
使用CMAQ模型值 $M(s)$ 将站点分为三个层级：
$$
\text{Level}_j = \begin{cases}
\text{low} & \text{if } M(s) < Q_{33} \\
\text{medium} & \text{if } Q_{33} \leq M(s) < Q_{67} \\
\text{high} & \text{if } M(s) \geq Q_{67}
\end{cases}
$$
其中 $Q_{33}$ 和 $Q_{67}$ 是模型值的33%和67%分位数。

### 步骤2：层级偏差建模
每个层级独立估计偏差：
$$
\hat{b}_j = \frac{\sum_{i \in \text{Level}_j} w_i (O(s_i) - M(s_i))}{\sum_{i \in \text{Level}_j} w_i}
$$
权重 $w_i = 1/d_i^2$ 基于距目标点的距离。

### 步骤3：平滑过渡
为避免分层边界不连续，使用软过渡：
$$
\alpha(M) = \text{sigmoid}\left(\frac{M - Q_{50}}{\sigma}\right)
$$
其中 $Q_{50}$ 是中位数，$\sigma = (Q_{67} - Q_{33})/4$。

### 步骤4：融合预测
$$
P_{CR-ABC}(s_0) = M(s_0) + \alpha(M(s_0)) \cdot \hat{b}_{\text{high}} + (1-\alpha(M(s_0))) \cdot \hat{b}_{\text{low}}
$$
当 $\alpha \approx 0$ 时使用低层偏差，$\alpha \approx 1$ 时使用高层偏差。

## 关键创新点
1. **非线性偏差建模**：捕捉模型偏差与浓度的非线性关系
2. **分层独立校正**：不同污染程度对应不同偏差模式
3. **软过渡避免突变**：sigmoid过渡确保空间平滑

## 与aVNA比较
| 特性 | aVNA（线性偏差） | CR-ABC（分层偏差） |
|------|----------------|-------------------|
| 偏差结构 | 全局线性 | 浓度依赖非线性 |
| 分层处理 | 无 | 三层独立建模 |
| 边界连续性 | 自然连续 | sigmoid软过渡 |

## 预期改进
R²提升 ≥ 0.01（相比aVNA）

## 参数清单
| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| n_levels | int | 3 | 浓度分层数 |
| percentile_low | float | 33.0 | 低层分位数阈值 |
| percentile_high | float | 67.0 | 高层分位数阈值 |
| transition_width | float | 0.25 | sigmoid过渡宽度（标准化单位） |
| k | int | 30 | 近邻数量 |
| power | float | -2 | 距离权重指数 |

## 方法指纹
MD5: cr_abc_concentration_regime_v1

## 输入输出格式
- 输入：监测数据CSV + CMAQ netCDF + 站点坐标
- 输出：融合网格PM2.5
- 支持十折空间交叉验证
