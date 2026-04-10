# 创新方法指令

## 方法名称
Spatial Leave-One-Out Cross-Validation Adaptive Kriging (SLOOCV-AK) - 空间留一交叉验证自适应克里金

## 创新核心
现有eVNA/aVNA使用固定k近邻和固定变异函数参数。SLOOCV-AK通过空间留一交叉验证自适应选择最优k值和变异函数参数，使每个局部区域的预测精度最优。

## 核心公式

### 步骤1：定义局部误差准则
对每个候选参数组合 $(\hat{a}, \hat{c}_0, \hat{c}, k)$，空间LOOCV误差：
$$
\text{LOOCV\_error} = \frac{1}{n_{\text{train}}} \sum_{i=1}^{n_{\text{train}}} (O(s_i) - \hat{O}^{-i}(s_i))^2
$$
其中 $\hat{O}^{-i}(s_i)$ 是剔除站点i后对站点i的预测值。

### 步骤2：参数搜索空间
- $k \in \{10, 15, 20, 30, 40, 50\}$
- 变异函数变程 $a \in \{20, 50, 100, 150\}$ km
- 块金效应 $c_0 \in \{0.1, 0.5, 1.0\} \times \text{Var}(R)$

### 步骤3：自适应k选择
对每个目标点，使用局部参数搜索：
$$
k^*(s_0) = \arg\min_k \text{LOOCV\_error}(k, a^*(k), c_0^*(k), c^*(k))
$$
其中内层优化对每个k找到最优变异函数参数。

### 步骤4：残差克里金预测
使用最优参数进行克里金：
$$
\hat{R}(s_0) = \sum_{i \in \mathcal{N}_{k^*}(s_0)} \lambda_i R(s_i)
$$
$$
P_{\text{SLOOCV-AK}}(s_0) = M(s_0) + \hat{R}(s_0)
$$

## 关键创新点
1. **空间LOOCV**：替代普通交叉验证，保持空间自相关性不被破坏
2. **自适应k**：每个网格点用最优近邻数，而非全局固定值
3. **局部参数优化**：变异函数参数在局部区域内优化

## 与VG-VNA比较
| 特性 | VG-VNA | SLOOCV-AK |
|------|--------|-----------|
| k近邻 | 全局固定 | 每个点自适应最优 |
| 变异函数参数 | 全局拟合 | 局部优化 |
| 参数选择 | 手动设定 | 数据驱动LOOCV |

## 预期改进
R²提升 ≥ 0.01（相比VG-VNA和eVNA/aVNA）

## 参数清单
| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| k_candidates | list | [10,15,20,30,40,50] | 候选近邻数 |
| range_candidates | list | [20,50,100,150] | 候选变程（km） |
| nugget_factors | list | [0.1,0.5,1.0] | 块金效应因子 |
| variogram_model | str | 'exponential' | 变异函数模型 |
| cv_strategy | str | 'spatial_loocv' | 交叉验证策略 |

## 方法指纹
MD5: sloocv_ak_spatial_adaptive_kriging_v1

## 输入输出格式
- 输入：监测数据CSV + CMAQ netCDF + 站点坐标
- 输出：融合网格PM2.5 + 局部不确定性
- 支持十折空间交叉验证
