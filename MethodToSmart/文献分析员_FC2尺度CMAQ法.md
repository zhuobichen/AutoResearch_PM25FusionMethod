# 【可执行方法规范】

## 方法名称
FC2 - 尺度CMAQ融合法 (Friberg FC2 Scaled CMAQ Method)

## 文献来源
- 论文标题: "Method for Fusing Observational Data and Chemical Transport Model Simulations To Estimate Spatiotemporally Resolved Ambient Air Pollution"
- 作者: Mariel D. Friberg et al.
- 期刊: Environmental Science & Technology, 2016, 50, 3695-3705
- DOI: 10.1021/acs.est.5b05134

## 核心公式

### FC2 融合公式 (方程3):
$$
FC_2(s,t) = CMAQ(s,t) \times \frac{\beta \times \overline{FC}(s)}{CMAQ_{annual}(s)} \times \beta_{season}(t)
$$

其中:
- $CMAQ(s,t)$ = 网格(s,t)处的CMAQ日值
- $\overline{FC}(s)$ = 校正后的年均CMAQ场
- $CMAQ_{annual}(s)$ = CMAQ年均值
- $\beta_{season}(t)$ = 季节校正因子

### 季节校正因子 (方程4):
$$
\beta_{season}(t) = 1 + A \times \cos\left[\frac{2\pi}{365.25}(t - t_{max})\right]
$$

其中:
- $A$ = 季节振幅参数
- $t_{max}$ = 峰值校正日期（一年中的天数）
- 365.25 = 一年天数

### 年均值场回归 (方程2):
$$
\overline{FC}(s) = \alpha_{year} \times \beta \times CMAQ_{year}(s)
$$
- $\alpha_{year}$: 年度回归截距
- $\beta$: 全局斜率参数（部分污染物β=1）

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| alpha | float | data-fitted | 年均回归截距 |
| beta | float | 1.0 | 斜率参数（部分污染物固定为1） |
| A | float | data-fitted | 季节振幅 |
| t_max | int | data-fitted | 峰值日期 |
| seasonal_correction | bool | True | 是否应用季节校正 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| CMAQ日场 | array | (n_grid, n_time) | μg/m³ |
| CMAQ年均场 | array | (n_grid,) | μg/m³ |
| 监测站点观测值 | array | (n_obs, n_time) | μg/m³ |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 融合网格场 | array | μg/m³ |

## 实现步骤

1. **计算CMAQ年均场**: 对CMAQ日场进行时间平均
2. **年度偏差校正**: 使用线性回归建立 $OBS_{annual,m}$ 与 $CMAQ_{annual,m}$ 的关系，获得α和β参数
3. **季节偏差校正**: 使用正弦函数拟合观测与CMAQ之间的季节性差异，获得A和t_max
4. **尺度化CMAQ**: 将CMAQ日场乘以年度校正因子和季节校正因子

## 方法特点
- 不依赖空间稀疏的观测网络
- 预测的时空变化依赖于每日气象和排放
- 误差与到观测的距离无关
- 主要受CMAQ模拟准确性限制

## 随机性
- [x] 否（确定性方法）

## 方法指纹
MD5: fc2_scaled_cmaq_method

## 实现检查清单
- [x] 核心公式已验证
- [x] 季节校正已实现
- [x] 年度回归已实现
