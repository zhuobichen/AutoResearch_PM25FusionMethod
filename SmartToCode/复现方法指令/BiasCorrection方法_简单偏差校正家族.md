# 【可执行方法规范】

## 方法名称
Bias Correction (BC) - 偏差校正法家族

## 文献来源
多个论文均有描述，这里综合整理：
- 《融合观测数据与化学传输模型模拟以估算时空分辨率的环境空气污染的方法》
- 《空气污染现场估算方法的交叉比较与评估》

## 核心公式

### 1. Mean BC（均值偏差校正）
$$
P_{MeanBC}(s_0) = M(s_0) + \bar{B}
$$
其中 $\bar{B} = \frac{1}{n} \sum_i [O(s_i) - M(s_i)]$

### 2. Spatial BC（空间偏差校正）
$$
P_{SpaBC}(s_0) = M(s_0) + \hat{B}(s_0)
$$
其中 $\hat{B}(s_0)$ 是偏差的空间插值（如IDW）
$$
\hat{B}(s_0) = \sum_i w_i \cdot [O(s_i) - M(s_i)]
$$

### 3. Scaling BC（缩放偏差校正）
$$
P_{ScaleBC}(s_0) = M(s_0) \cdot \bar{r}
$$
其中 $\bar{r} = \frac{1}{n} \sum_i \frac{O(s_i)}{M(s_i)}$

### 4. Linear BC（线性偏差校正）
$$
P_{LinBC}(s_0) = a + b \cdot M(s_0)
$$
其中 $O = a + b \cdot M + \epsilon$ (OLS拟合)

## 参数清单
| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| method | str | 'spatial' | 'mean', 'spatial', 'scale', 'linear' |
| k | int | 30 | 近邻数量（spatial方法）|
| power | float | -2 | 距离权重指数 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 监测站点坐标 | array | (n, 2) | 度 |
| 监测站点观测值 | array | (n,) | μg/m³ |
| CMAQ模型值（站点） | array | (n,) | μg/m³ |
| CMAQ模型值（网格） | array | (n_grid,) | μg/m³ |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 融合网格 | array | μg/m³ |

## 实现步骤

### Mean BC
1. 计算平均偏差：$\bar{B} = \text{mean}(O - M)$
2. 融合：$P = M + \bar{B}$

### Spatial BC
1. 对每个站点计算偏差：$B_i = O_i - M_i$
2. 对网格点插值偏差：$\hat{B}(s_0) = \sum_i w_i \cdot B_i$
3. 融合：$P(s_0) = M(s_0) + \hat{B}(s_0)$

### Scale BC
1. 计算平均比率：$\bar{r} = \text{mean}(O / M)$
2. 融合：$P = M \cdot \bar{r}$

### Linear BC
1. OLS拟合：$O = a + b \cdot M + \epsilon$
2. 融合：$P = a + b \cdot M$

## 随机性
- [x] 否（确定性方法）

## 方法指纹
MD5: bc_method_fingerprint_v1

## 实现检查清单
- [x] 核心公式已验证
- [ ] 边界条件已处理
- [ ] 单元测试通过
