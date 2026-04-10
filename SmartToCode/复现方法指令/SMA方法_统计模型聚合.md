# 【可执行方法规范】

## 方法名称
Statistical Model Aggregation (SMA) - 统计模型聚合

## 文献来源
- 论文标题：《融合观测数据与化学传输模型模拟以估算时空分辨率的环境空气污染的方法》
- 作者：Shao et al.
- 关键章节：Section 2.2.2

## 核心公式
首先通过线性回归建立观测与模型的统计关系：
$$
O(s_i) = a + b \cdot M(s_i) + \epsilon(s_i)
$$

然后将模型输出映射到观测尺度：
$$
P_{SMA}(s_0) = \hat{a} + \hat{b} \cdot M(s_0)
$$

其中 $\hat{a}, \hat{b}$ 通过OLS估计：
$$
\hat{b} = \frac{\sum_i (M(s_i) - \bar{M})(O(s_i) - \bar{O})}{\sum_i (M(s_i) - \bar{M})^2}
$$
$$
\hat{a} = \bar{O} - \hat{b} \cdot \bar{M}
$$

## 参数清单
| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| regression_type | str | 'linear' | 'linear'或'polynomial' |
| poly_degree | int | 1 | 多项式阶数（仅当regression_type='polynomial'时）|
| robust | bool | False | 是否使用稳健回归（Huber）|

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
1. 输入：$(O_i, M_i), i=1,...,n$
2. 使用OLS拟合：$O = a + b \cdot M + \epsilon$
3. 得到参数估计 $\hat{a}, \hat{b}$
4. 对网格点预测：$P_{SMA} = \hat{a} + \hat{b} \cdot M_{grid}$
5. 残差分析：$R_i = O_i - (\hat{a} + \hat{b} \cdot M_i)$

## 多项式SMA变体
$$
O(s_i) = a_0 + a_1 \cdot M(s_i) + a_2 \cdot M^2(s_i) + ... + a_k \cdot M^k(s_i) + \epsilon(s_i)
$$
$$
P_{SMA-poly}(s_0) = \hat{a}_0 + \hat{a}_1 \cdot M(s_0) + \hat{a}_2 \cdot M^2(s_0) + ...
$$

## 随机性
- [x] 否（确定性方法，OLS有解析解）

## 方法指纹
MD5: sma_method_fingerprint_v1

## 实现检查清单
- [x] 核心公式已验证
- [ ] 边界条件已处理
- [ ] 单元测试通过
