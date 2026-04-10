# 【可执行方法规范】

## 方法名称
FC_opt - 优化加权融合法 (Friberg Optimized Fusion Method)

## 文献来源
- 论文标题: "Method for Fusing Observational Data and Chemical Transport Model Simulations To Estimate Spatiotemporally Resolved Ambient Air Pollution"
- 作者: Mariel D. Friberg et al.
- 期刊: Environmental Science & Technology, 2016, 50, 3695-3705
- DOI: 10.1021/acs.est.5b05134

## 核心公式

### 加权因子 (方程8):
$$
W(s,t) = \frac{[R_1(s,t) - R_2] \times R_1(s,t)}{[R_1(s,t) - R_2]^2 + [R_1(s,t) \times (1 - R_1(s,t))] \times (1 - W_{min})}
$$

其中:
- $R_1(s,t)$ = FC1方法的估计时空相关性
- $R_2$ = CMAQ方法的时间相关性（不依赖距离）
- $W_{min}$ = 最小权重（通常为0）

### 时空相关性 R1 (方程6):
$$
R_1(s,t) \approx R_{coll} + (1 - R_{coll}) \times e^{-x_{st}/r}
$$
其中 $x_{st}$ = 网格(s,t)到最近观测的距离

### CMAQ相关性 R2 (方程7):
$$
R_2 \approx \frac{1}{N}\sum_{m=1}^{N} corr(OBS_m(t), CMAQ_m(t))
$$
即观测与CMAQ在所有监测站的时间相关性均值。

### 最终融合公式 (方程9):
$$
FC_{opt}(s,t) = W(s,t) \times FC_1(s,t) + (1 - W(s,t)) \times FC_2(s,t)
$$

### 加权融合相关性 (方程10):
$$
R_{opt}(s,t) = W(s,t) \times R_1(s,t) + (1 - W(s,t)) \times R_2 \quad (当R_1 > R_2)
$$
$$
R_{opt} = R_2 \quad (否则)
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| W_min | float | 0.0 | 最小权重 |
| R_coll | float | data-fitted | 截距相关系数（仪器误差） |
| r | float | data-fitted | 距离尺度参数 |
| R_cmaq | float | data-fitted | CMAQ-观测时间相关系数 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| FC1结果 | array | (n_grid, n_time) | μg/m³ |
| FC2结果 | array | (n_grid, n_time) | μg/m³ |
| 监测站点坐标 | array | (n_obs, 2) | 度 |
| 网格坐标 | array | (n_grid, 2) | 度 |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 优化融合场 | array | μg/m³ |
| 权重场W | array | (n_grid, n_time) | 无量纲 |

## 实现步骤

1. **计算R1**: 对每个网格点，根据到最近观测的距离计算R1
2. **计算R2**: 计算观测与CMAQ在监测站的时间相关性均值
3. **计算权重W**: 根据方程8计算空间变化的权重
4. **融合**: 对FC1和FC2进行加权平均

## 加权特性
- 当 $R_1 = 1$ 时，$W = 1$（完全使用FC1）
- 当 $R_1 = R_2$ 时，$W = 0.5$（等权重）
- 当 $R_1 = 0$ 时，$W = 0$（完全使用FC2）

## 方法优势
- 在观测附近依赖FC1（更准确）
- 在远离观测处依赖FC2（不依赖观测距离）
- 提供时空变化的预测不确定性

## 随机性
- [x] 否（确定性方法）

## 方法指纹
MD5: fcopt_optimized_fusion_method

## 实现检查清单
- [x] 核心公式已验证
- [x] 权重计算已实现
- [x] 加权融合已实现
