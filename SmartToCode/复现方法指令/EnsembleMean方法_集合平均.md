# 【可执行方法规范】

## 方法名称
Ensemble Mean (EM) - 集合平均法

## 文献来源
- 《通过融合观测数据与集合化学传输模型模拟结果，实现时空分辨率下的环境颗粒物浓度估算》
- 关键章节：Section 2.2 集合平均融合方法

## 核心公式
当有多个CMAQ集合成员（ensemble）时：

### 集合平均
$$
\bar{M}(s) = \frac{1}{N_e} \sum_{k=1}^{N_e} M_k(s)
$$

### 带偏差校正的集合平均
$$
P_{EM-BC}(s_0) = \bar{M}(s_0) + \sum_i w_i \cdot [O(s_i) - \bar{M}(s_i)]
$$

或使用缩放形式：
$$
P_{EM-SC}(s_0) = \bar{M}(s_0) \cdot \frac{\sum_i w_i \cdot O(s_i)}{\sum_i w_i \cdot \bar{M}(s_i)}
$$

### 权重集合平均
$$
P_{WEM}(s_0) = \sum_{k=1}^{N_e} w_k \cdot M_k(s_0)
$$
其中权重 $w_k$ 基于历史表现确定

## 参数清单
| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| method | str | 'mean' | 'mean', 'weighted', 'bias_corrected' |
| weights | array | None | 集合成员权重（若为None则等权重）|
| k | int | 30 | 近邻数量 |
| power | float | -2 | 距离权重指数 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 监测站点坐标 | array | (n, 2) | 度 |
| 监测站点观测值 | array | (n,) | μg/m³ |
| CMAQ集合成员值（站点） | array | (n, N_e) | μg/m³ |
| CMAQ集合成员值（网格） | array | (n_grid, N_e) | μg/m³ |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 融合网格 | array | μg/m³ |

## 实现步骤
1. **计算集合平均**：$\bar{M} = \text{mean}(M_1, ..., M_{N_e})$
2. **计算站点偏差**（若使用BC）：$B_i = O_i - \bar{M}_i$
3. **插值偏差**：$\hat{B}(s_0) = \sum_i w_i \cdot B_i$
4. **融合**：$P(s_0) = \bar{M}(s_0) + \hat{B}(s_0)$

## 与单成员CMAQ的关系
- 当 $N_e = 1$ 时，EM退化为标准的 aVNA（加法形式）
- EM利用了集合预报的不确定性信息

## 随机性
- [x] 否（确定性方法）

## 方法指纹
MD5: em_method_fingerprint_v1

## 实现检查清单
- [x] 核心公式已验证
- [ ] 边界条件已处理
- [ ] 单元测试通过
