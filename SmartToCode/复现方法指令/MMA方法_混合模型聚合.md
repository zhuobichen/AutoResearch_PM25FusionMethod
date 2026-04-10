# 【可执行方法规范】

## 方法名称
Mixed Model Aggregation (MMA) - 混合模型聚合

## 文献来源
- 论文标题：《融合观测数据与化学传输模型模拟以估算时空分辨率的环境空气污染的方法》
- 作者：Shao et al.
- 关键章节：Section 2.2.3

## 核心公式
MMA结合了SMA的全局统计校正和局部空间插值的优点：

$$
P_{MMA}(s_0) = \beta \cdot [\hat{a} + \hat{b} \cdot M(s_0)] + (1-\beta) \cdot [M(s_0) + \sum_i w_i \cdot (O(s_i) - M(s_i))]
$$

其中：
- $\hat{a}, \hat{b}$ 为OLS回归系数
- $w_i$ 为距离权重（如VNA中的反距离加权）
- $\beta$ 为混合参数，0≤β≤1

等价形式：
$$
P_{MMA}(s_0) = M(s_0) + \beta \cdot [\hat{a} + \hat{b} \cdot M(s_0) - M(s_0)] + (1-\beta) \cdot \sum_i w_i \cdot [O(s_i) - M(s_i)]
$$

简化形式（当β=0时退化为aVNA）：
$$
P_{MMA}(s_0) = M(s_0) + \sum_i w_i \cdot [O(s_i) - M(s_i)]
$$

## 参数清单
| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| beta | float | 0.5 | 混合参数（0=SMA主导，1=aVNA主导）|
| k | int | 30 | 近邻数量 |
| power | float | -2 | 距离权重指数 |
| alpha | float | 0.05 | 统计显著性水平（用于置信区间）|

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
1. **步骤1**：OLS回归拟合 $O = a + b \cdot M + \epsilon$
2. **步骤2**：计算局部偏差插值 $D_{VNA}(s_0) = \sum_i w_i \cdot [O(s_i) - M(s_i)]$
3. **步骤3**：计算全局偏差校正 $D_{SMA}(s_0) = \hat{a} + \hat{b} \cdot M(s_0) - M(s_0) = \hat{a} + (\hat{b}-1) \cdot M(s_0)$
4. **步骤4**：混合融合 $P_{MMA}(s_0) = M(s_0) + \beta \cdot D_{SMA}(s_0) + (1-\beta) \cdot D_{VNA}(s_0)$

## 优化beta的方法
通过交叉验证选择最优β：
$$
\beta^* = \arg\min_\beta \text{CVError}(\beta)
$$

## 随机性
- [x] 否（确定性方法）

## 方法指纹
MD5: mma_method_fingerprint_v1

## 实现检查清单
- [x] 核心公式已验证
- [ ] 边界条件已处理
- [ ] 单元测试通过
