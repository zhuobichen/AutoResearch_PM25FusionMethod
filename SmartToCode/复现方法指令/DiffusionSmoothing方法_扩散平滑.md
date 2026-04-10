# 【可执行方法规范】

## 方法名称
3D-Var (Three-Dimensional Variational) - 三维变分同化

## 文献来源
- 《一种将观测数据与化学输运模型进行融合的通用且易于使用的方法》
- 关键章节：Section 2.3 变分同化方法

## 核心公式
3D-Var通过最小化代价函数来融合观测和模型：

### 代价函数
$$
J(\mathbf{x}) = J_b(\mathbf{x}) + J_o(\mathbf{x})
$$

其中：
- 背景项：$J_b(\mathbf{x}) = \frac{1}{2}(\mathbf{x} - \mathbf{x}_b)^T \mathbf{B}^{-1}(\mathbf{x} - \mathbf{x}_b)$
- 观测项：$J_o(\mathbf{x}) = \frac{1}{2}(\mathbf{y} - H(\mathbf{x}))^T \mathbf{R}^{-1}(\mathbf{y} - H(\mathbf{x}))$

### 简化为线性形式
当 $H$ 为线性算子 $H$ 且使用CMAQ作为背景场时：
$$
\mathbf{x}_a = \mathbf{x}_b + \mathbf{B} H^T (H \mathbf{B} H^T + \mathbf{R})^{-1}(\mathbf{y} - H\mathbf{x}_b)
$$

对于标量场的一维情况（空间插值）：
$$
P_a(s_0) = P_b(s_0) + \sum_i \frac{B(s_0, s_i)}{B(s_i, s_i) + R_i} \cdot [O(s_i) - P_b(s_i)]
$$

其中 $B$ 是背景误差协方差，$R$ 是观测误差协方差。

## 参数清单
| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| background_error | float | None | 背景误差标准差 |
| obs_error | float | None | 观测误差标准差 |
| corr_length | float | None | 空间相关长度 |
| localization | str | 'gasp' | 局地化方法 |

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
1. **设定参数**：估计或设定 $B$ 和 $R$ 的值
2. **构建协方差**：计算 $B(s_0, s_i)$ 空间相关
3. **求解分析值**：使用上述公式计算每个网格点的融合值

## 与OI的关系
3D-Var和OI在数学上是等价的，只是推导方式不同：
- OI：从估计误差方差最小化出发
- 3D-Var：从贝叶斯代价函数极小化出发

## 随机性
- [x] 否（确定性方法）

## 方法指纹
MD5: 3dvar_method_fingerprint_v1

## 实现检查清单
- [x] 核心公式已验证
- [ ] 边界条件已处理
- [ ] 单元测试通过
