# 【可执行方法规范】

## 方法名称
Observation Deviation Index (ODI) - 观测偏差指示器法

## 文献来源
- 论文标题：《一种将观测数据与多模型输出相结合的新方法（M3Fusion v1）》
- 关键章节：Section 2.2.2 ODI方法

## 核心公式
ODI方法基于模型与观测的偏差程度来调整融合权重：

### 偏差指示器
$$
DI(s_i) = \frac{O(s_i) - M(s_i)}{M(s_i)} = \frac{B(s_i)}{M(s_i)}
$$

### 归一化偏差指示器
$$
NDI(s_i) = \frac{DI(s_i) - \mu_{DI}}{\sigma_{DI}}
$$

### 空间平滑NDI
$$
\widehat{NDI}(s_0) = \sum_i w_i \cdot NDI(s_i)
$$

### 融合预测
$$
P_{ODI}(s_0) = M(s_0) \cdot [1 + \gamma \cdot \widehat{NDI}(s_0)]
$$

或等价形式：
$$
P_{ODI}(s_0) = M(s_0) + \gamma \cdot M(s_0) \cdot \widehat{NDI}(s_0)
$$

其中 $\gamma$ 是缩放因子，控制NDI对融合结果的影响程度。

## 参数清单
| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| gamma | float | 1.0 | NDI缩放因子 |
| k | int | 30 | 近邻数量 |
| power | float | -2 | 距离权重指数 |
| normalize | bool | True | 是否归一化DI |

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
1. **计算偏差指示器**：$DI_i = (O_i - M_i) / M_i$
2. **归一化**：计算 $\mu_{DI}, \sigma_{DI}$，得 $NDI_i = (DI_i - \mu_{DI}) / \sigma_{DI}$
3. **空间插值**：对网格点 $s_0$，计算 $\widehat{NDI}(s_0) = \sum_i w_i \cdot NDI_i$
4. **融合**：$P(s_0) = M(s_0) \cdot [1 + \gamma \cdot \widehat{NDI}(s_0)]$

## 与eVNA的关系
当 $\gamma = 1$ 且使用正确的归一化时，ODI与eVNA有相似结构，但ODI：
- 对偏差进行了归一化（除以模型值）
- 允许 $\gamma$ 参数调节影响程度

## 随机性
- [x] 否（确定性方法）

## 方法指纹
MD5: odi_method_fingerprint_v1

## 实现检查清单
- [x] 核心公式已验证
- [ ] 边界条件已处理
- [ ] 单元测试通过
