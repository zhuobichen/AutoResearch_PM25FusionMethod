# 【可执行方法规范】

## 方法名称
Observation Model Aggregation (OMA) - 观测模型聚合

## 文献来源
- 论文标题：《融合观测数据与化学传输模型模拟以估算时空分辨率的环境空气污染的方法》
- 作者：Shao et al.
- 关键章节：Section 2.2.1

## 核心公式
$$
P_{OMA}(s_0) = \alpha \cdot O(s_0) + (1-\alpha) \cdot M(s_0)
$$
其中 $\alpha$ 为优化权重，通过最小化以下目标函数获得：
$$
\alpha^* = \arg\min_\alpha \sum_i [O(s_i) - \alpha \cdot O(s_i) - (1-\alpha) \cdot M(s_i)]^2
$$

等价形式（偏差校正形式）：
$$
P_{OMA}(s_0) = M(s_0) + \alpha \cdot [O(s_i) - M(s_i)]
$$

## 参数清单
| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| alpha | float | 0.5 | 融合权重（0=OBS完全主导，1=CMAQ完全主导）|
| method | str | 'global' | 'global'或'local'，全局权重或局部权重 |

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
1. 计算站点偏差：$D(s_i) = O(s_i) - M(s_i)$
2. 优化权重 $\alpha$：最小化 $\sum_i [D(s_i) - \alpha \cdot D(s_i)]^2$
3. 对于网格点：$P_{OMA}(s_0) = M(s_0) + \alpha \cdot \bar{D}$ （全局OMA）
4. 若用局部版本：对每个网格点取最近k个站点的加权偏差

## 局部OMA变体（VNA-like）
$$
P_{OMA}(s_0) = M(s_0) + \sum_i w_i \cdot [O(s_i) - M(s_i)]
$$
其中 $w_i = \frac{1/d_i^p}{\sum_j 1/d_j^p}$

## 随机性
- [x] 否（确定性方法）

## 方法指纹
MD5: oma_method_fingerprint_v1

## 实现检查清单
- [x] 核心公式已验证
- [ ] 边界条件已处理
- [ ] 单元测试通过
