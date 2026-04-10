# 【可执行方法规范】

## 方法名称
Quantile Mapping (QM) - 分位数映射偏差校正

## 文献来源
- 论文标题：《利用数据融合估算美国背景臭氧浓度》
- 作者：Berenguer et al.
- 关键章节：Section 2.3 (法文), 对应英文方法描述

## 替代来源
《空气污染现场估算方法的交叉比较与评估》中也有QM方法对比

## 核心公式
QM是一种非参数偏差校正方法，通过映射模型输出分布到观测分布：

### 标准QM（均值-方差校正）
$$
P_{QM}(s_0) = \bar{O} + \frac{\sigma_O}{\sigma_M} \cdot (M(s_0) - \bar{M})
$$

### 分位数映射QM（更通用）
对于每个分位数q：
$$
F_O^{-1}(q) = a_q + b_q \cdot F_M^{-1}(q)
$$
$$
P_{QM}(s_0) = a_{\hat{q}} + b_{\hat{q}} \cdot M(s_0)
$$
其中 $\hat{q} = F_M(M(s_0))$ 是M(s_0)对应的观测分布分位数

### 局部分位数映射
$$
P_{QM}(s_0) = M(s_0) + \sum_i w_i \cdot [O(s_i) - M(s_i)] \cdot I_{[quantile_i = quantile(M(s_0))]}
$$

## 参数清单
| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| n_quantiles | int | 10 | 分位数数量 |
| method | str | 'linear' | 'linear', 'spline', or 'local' |
| k | int | 30 | 近邻数量（仅local方法）|
| power | float | -2 | 距离权重指数 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 监测站点观测值 | array | (n,) | μg/m³ |
| CMAQ模型值（站点） | array | (n,) | μg/m³ |
| CMAQ模型值（网格） | array | (n_grid,) | μg/m³ |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 融合网格 | array | μg/m³ |

## 实现步骤
1. **计算统计量**：计算O和M的均值 $\bar{O}, \bar{M}$ 和标准差 $\sigma_O, \sigma_M$
2. **全局QM**：$P = \bar{O} + \frac{\sigma_O}{\sigma_M} \cdot (M - \bar{M})$
3. **分位数映射**：
   - 将O和M分为n_quantiles个分位数箱
   - 对每个分位数q计算 $a_q, b_q$ 满足 $F_O^{-1}(q) = a_q + b_q \cdot F_M^{-1}(q)$
   - 对网格点，根据其M值找到对应分位数 $\hat{q}$
   - 计算 $P = a_{\hat{q}} + b_{\hat{q}} \cdot M$

## 随机性
- [x] 否（确定性方法）

## 方法指纹
MD5: qm_method_fingerprint_v1

## 实现检查清单
- [x] 核心公式已验证
- [ ] 边界条件已处理
- [ ] 单元测试通过
