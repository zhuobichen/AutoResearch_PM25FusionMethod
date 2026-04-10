# 【可执行方法规范】

## 方法名称
Spatial Kriging Bias Correction (SK-BC) - 空间克里金偏差校正

## 文献来源
- 论文标题：《基于时空克里金模型估算北京地区日均PM2.5暴露量》
- 作者：北京大学环境科学与工程学院
- 关键章节：Section 2.2 (时空克里金方法)

## 核心公式
该方法使用克里金插值来估计空间变化的偏差：

### 偏差计算
$$
B(s_i) = O(s_i) - M(s_i)
$$

### 克里金插值偏差
$$
\hat{B}(s_0) = \sum_{i=1}^{n} \lambda_i(s_0) \cdot B(s_i)
$$

其中权重 $\lambda_i(s_0)$ 通过克里金系统求解：
$$
\sum_j \lambda_j(s_0) \cdot C(s_i, s_j) = C(s_i, s_0)
$$
其中 $C(h)$ 是偏差场的半变异函数/协方差函数。

### 融合结果
$$
P_{SK-BC}(s_0) = M(s_0) + \hat{B}(s_0)
$$

## 参数清单
| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| variogram_model | str | 'spherical' | 半变异函数模型 |
| range | float | None | 变异程（自动估计）|
| sill | float | None | 块金值（自动估计）|
| nugget | float | 0 | 块金效应 |
| kriging_type | str | 'ordinary' | 'ordinary'或'simple' |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 监测站点坐标 | array | (n, 2) | 度 |
| 监测站点观测值 | array | (n,) | μg/m³ |
| CMAQ模型值（站点） | array | (n,) | μg/m³ |
| CMAQ模型值（网格） | array | (n_grid,) | μg/m³ |
| 网格点坐标 | array | (n_grid, 2) | 度 |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 融合网格 | array | μg/m³ |

## 实现步骤
1. **计算站点偏差**：$B_i = O_i - M_i$
2. **拟合变异函数**：使用监测站点的偏差数据拟合半变异函数 $\gamma(h)$
3. **建立克里金系统**：对每个网格点 $s_0$，求解克里金权重
4. **克里金插值**：$\hat{B}(s_0) = \sum_i \lambda_i(s_0) \cdot B_i$
5. **融合**：$P(s_0) = M(s_0) + \hat{B}(s_0)$

## 变异函数模型
- Spherical: $\gamma(h) = c_0 + c \cdot [1.5h/a - 0.5(h/a)^3]$ for $h \leq a$
- Exponential: $\gamma(h) = c_0 + c \cdot [1 - \exp(-h/a)]$
- Gaussian: $\gamma(h) = c_0 + c \cdot [1 - \exp(-h^2/a^2)]$

## 随机性
- [x] 否（确定性方法）

## 方法指纹
MD5: skbc_method_fingerprint_v1

## 实现检查清单
- [x] 核心公式已验证
- [ ] 边界条件已处理
- [ ] 单元测试通过
