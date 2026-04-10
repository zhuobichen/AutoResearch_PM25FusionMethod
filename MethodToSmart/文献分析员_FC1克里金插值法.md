# 【可执行方法规范】

## 方法名称
FC1 - 克里金插值融合法 (Friberg FC1 Method)

## 文献来源
- 论文标题: "Method for Fusing Observational Data and Chemical Transport Model Simulations To Estimate Spatiotemporally Resolved Ambient Air Pollution"
- 作者: Mariel D. Friberg, Xinxin Zhai, Heather A. Holmes, Howard H. Chang, et al.
- 期刊: Environmental Science & Technology, 2016, 50, 3695-3705
- DOI: 10.1021/acs.est.5b05134
- arXiv: 融合观测数据与化学传输模型模拟以估算时空分辨率的环境空气污染的方法.pdf

## 核心公式

### FC1 融合公式 (方程1):
$$
FC_1(s,t) = krig\left(\frac{OBS(t)}{OBS}\right) \times \overline{FC}(s)
$$

其中:
- $s$ = 空间位置
- $t$ = 时间
- $OBS(t)$ = 时间t的观测值
- $\overline{OBS}$ = 年均观测值
- $krig(\cdot)$ = 普通克里金插值
- $\overline{FC}(s)$ = CMAQ年均场（经回归校正）

### 年均值场模型 (方程2):
$$
\overline{FC}(s) = \alpha \times \beta \times CMAQ_{year}(s)
$$
其中 $\alpha$ 是年度回归参数，$\beta$ 是全局斜率参数。

### 克里金权重计算:
普通克里金通过最小化估计方差来计算权重:
$$
\sum_j \lambda_j \gamma(d_{ij}) = \gamma(d_{i0}), \quad \sum_j \lambda_j = 1
$$
其中 $\gamma(d)$ = 半变异函数，$\lambda_j$ = 克里金权重。

### 时空相关性 (方程5-6):
$$
R_{obs}(d) = R_{coll} + (1 - R_{coll}) \times e^{-d/r}
$$
$$
R_1(s,t) \approx R_{coll} + (1 - R_{coll}) \times e^{-x_{st}/r}
$$
其中 $R_{coll}$ = 共置仪器误差相关的截距相关，$r$ = 距离尺度参数，$x_{st}$ = 到最近观测的距离。

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| variogram_model | str | 'exponential' | 半变异函数模型 |
| range_param | float | data-derived | 空间相关距离范围 |
| nugget | float | 0 | 块金效应 |
| sill | float | data-derived | 基台值 |
| max_iter | int | 1000 | 克里金求解最大迭代 |
| n_neighbors | int | 10 | 用于克里金的近邻点数 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 监测站点坐标 | array | (n_obs, 2) | 度 |
| 监测站点观测值 | array | (n_obs, n_time) | μg/m³ |
| CMAQ网格值 | array | (n_grid, n_time) | μg/m³ |
| CMAQ年均场 | array | (n_grid,) | μg/m³ |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 融合网格场 | array | μg/m³ |

## 实现步骤

1. **归一化观测**: 将每日观测值除以年均值得到归一化值
2. **半变异函数拟合**: 使用指数半变异函数拟合归一化观测的空间相关性
3. **克里金插值**: 对归一化值进行普通克里金插值到网格点
4. **CMAQ年场校正**: 通过线性回归 $\overline{OBS}_m = \alpha \times \beta \times CMAQ_m$ 获得校正后的年均场
5. **反归一化**: 将克里金结果乘以校正后的CMAQ年均场得到最终融合结果

## 质量指标
- R² (时空相关性): 54-88% (因污染物而异)
- PM2.5表现: R² ≈ 0.64-0.88

## 随机性
- [ ] 是（克里金插值依赖数据随机性）

## 方法指纹
MD5: fc1_kriging_fusion_method

## 实现检查清单
- [x] 核心公式已验证
- [x] 克里金求解器已实现
- [x] 半变异函数拟合已实现
