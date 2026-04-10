# 【可执行方法规范】

## 方法名称
GP-Downscaling - 高斯过程尺度分解法 (Gaussian Process Disaggregation)

## 文献来源
- 论文标题: "Multivariate disaggregation modeling of air pollutants: a case-study of PM2.5, PM10 and ozone prediction in Portugal and Italy"
- 作者: Fernando Rodriguez Avellaneda, Erick A. Chacon-Montalvan, Paula Moraga
- 机构: King Abdullah University of Science and Technology (KAUST)
- 日期: 2025年3月

## 核心公式

### 多变量线性协同区域化模型 (LMC) (方程1):
$$
W_m(s) = \sum_{k=1}^{K} a_{mk} U_k(s)
$$
其中:
- $W_m(s)$ = m维响应过程在位置s的值
- $U_k(s)$ = 独立潜在空间过程
- $a_{mk}$ = 载荷系数矩阵元素
- K = 潜在过程数量

### 嵌套结构 (方程9):
$$
W_1(s) = U_1(s)
$$
$$
W_2(s) = \lambda_{21} U_1(s) + U_2(s)
$$
$$
W_3(s) = \lambda_{31} U_1(s) + \lambda_{32} U_2(s) + U_3(s)
$$

### 网格平均到区域汇总 (方程8):
$$
U_j(R_i) = \frac{1}{|R_i|} \int_{R_i} U_j(s) ds \approx \sum_{k=1}^{G} A_{ik} U_{j,k}
$$
其中 $A_{ik} = |R_{ik}|/|R_i|$ 是投影矩阵元素

### 区域层级模型 (方程11):
$$
W_1(R_i) = \alpha_1 + \bar{z}_1(R_i) + e_1(i)
$$
$$
W_2(R_i) = \alpha_2 + \lambda_{21}\bar{z}_1(R_i) + \bar{z}_2(R_i) + e_2(i)
$$
$$
W_3(R_i) = \alpha_3 + \lambda_{31}\bar{z}_1(R_i) + \lambda_{32}\bar{z}_2(R_i) + \bar{z}_3(R_i) + e_3(i)
$$

### Matérn协方差函数:
$$
Cov(Z(s_i), Z(s_j)) = \frac{\sigma^2}{\Gamma(\nu) 2^{\nu-1}} \left(\frac{\sqrt{2\nu} \cdot d}{\rho}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu} \cdot d}{\rho}\right)
$$
其中 $d$ = 欧氏距离，$\rho$ = 空间尺度参数，$\nu$ = 平滑参数

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| n_latent | int | 3 | 潜在过程数量 |
| nu | float | 1.0 | Matérn平滑参数 |
| max_iter | int | 500 | MCMC或优化最大迭代 |
| n_chain | int | 3 | MCMC链数 |
| burn | int | 200 | 燃烧期 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| CAMS网格场 | array | (n_grid, n_time) | μg/m³ |
| 监测数据（聚合） | array | (n_region, n_time) | μg/m³ |
| 网格坐标 | array | (n_grid, 2) | 度 |
| 区域划分 | shapefile | - | - |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 降尺度网格场 | array | μg/m³ |
| 区域参数 | dict | - | - |

## 实现步骤

1. **构建区域划分**: 将研究区域划分为不规则或规则区域
2. **聚合CAMS数据**: 将高分辨率网格数据聚合到区域平均值
3. **拟合LMC模型**: 估计载荷矩阵A和潜在过程参数
4. **MCMC采样**: 对区域层级模型进行贝叶斯推断
5. **预测连续场**: 从拟合的模型预测任意位置的浓度值
6. **交叉验证**: 使用留一法或k折交叉验证评估

## 方法优势
- 保持多变量之间的相关性
- 提供不确定性量化
- 适用于不规则区域划分
- 可迁移到新区域

## 随机性
- [ ] 是（MCMC采样带随机种子）

## 方法指纹
MD5: gp_disaggregation_method

## 实现检查清单
- [x] 核心公式已验证
- [x] LMC模型已实现
- [x] MCMC采样已实现
- [x] 区域映射已实现
