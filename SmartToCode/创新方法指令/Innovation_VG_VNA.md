# 创新方法指令

## 方法名称
Variogram-Geometric VNA (VG-VNA) - 变异函数几何加权VNA

## 创新核心
用变异函数建模的空间相关性替代简单反距离权重，使权重反映真实空间相关结构。相比eVNA/aVNA的固定1/d²假设，变异函数权重基于数据驱动，可自适应不同尺度的空间变化。

## 核心公式

### 步骤1：计算残差
$$
R(s_i) = O(s_i) - M(s_i)
$$

### 步骤2：经验变异函数
$$
\gamma(h_k) = \frac{1}{2 N(h_k)} \sum_{i,j \in N(h_k)} (R(s_i) - R(s_j))^2
$$
其中 $h_k$ 是距离滞后，$N(h_k)$ 是距离在 $[h_k - \Delta h, h_k + \Delta h]$ 内的点对数。

### 步骤3：变异函数拟合（指数模型）
$$
\gamma(h) = c_0 + c \cdot \left(1 - e^{-\frac{h}{a}}\right)
$$
- $c_0$: 块金效应（nugget）
- $c$: 拱高（partial sill = sill - nugget）
- $a$: 变程（range）

### 步骤4：克里金权重
构建克里金矩阵：
$$
\Gamma_{ij} = \gamma(|s_i - s_j|), \quad \gamma_0 = [\gamma(|s_0-s_1|), ..., \gamma(|s_0-s_n|)]^T
$$
求解：$\lambda = \Gamma^{-1} \gamma_0$

### 步骤5：残差预测与融合
$$
\hat{R}(s_0) = \sum_{i=1}^{n} \lambda_i R(s_i)
$$
$$
P_{VG-VNA}(s_0) = M(s_0) + \hat{R}(s_0)
$$

## 关键创新点
1. **变异函数权重**：用数据驱动的空间相关结构替代固定1/d²假设
2. **残差克里金**：对CMAQ残差做真正的空间插值，而非简单IDW
3. **局部自适应**：对每个目标点单独构建克里金系统

## 与eVNA/aVNA比较
| 特性 | eVNA/aVNA | VG-VNA |
|------|----------|--------|
| 权重来源 | 固定1/d² | 数据驱动变异函数 |
| 空间相关性 | 隐式假设 | 显式建模 |
| 计算复杂度 | O(n) | O(n³) for matrix solve |

## 预期改进
R²提升 ≥ 0.01（相比eVNA/aVNA）

## 参数清单
| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| variogram_model | str | 'exponential' | 变异函数模型 |
| n_lags | int | 15 | 距离滞后数量 |
| lag_size | float | 5.0 | 滞后间距（km） |
| max_range | float | 200.0 | 最大变程（km） |
| k | int | 30 | 克里金近邻数 |

## 方法指纹
MD5: vg_vna_variogram_geometric_v1

## 输入输出格式
- 输入：监测数据CSV + CMAQ netCDF + 站点坐标
- 输出：融合网格PM2.5
- 支持十折空间交叉验证
