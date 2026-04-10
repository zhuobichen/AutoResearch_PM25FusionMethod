# 【可执行方法规范】

## 方法名称
Cokriging - 共克里金法 (Cokriging for Multi-variable Spatial Prediction)

## 文献来源
- 论文标题: "A cokriging based approach to reconstruct air pollution maps" (Singh et al., 2011)
- 方法: 使用主变量和辅助变量的空间相关性进行联合插值

## 核心公式

### 互协方差函数:
$$
C_{UV}(h) = Cov(U(s), V(s+h))
$$

### 共克里金估计:
$$
\hat{U}(s_0) = \sum_{i=1}^{n} \lambda_i^U U(s_i) + \sum_{j=1}^{m} \lambda_j^V V(s_j)
$$
权重通过最小化估计方差确定，约束为无偏条件:
$$
\sum_{i=1}^{n} \lambda_i^U = 1, \quad \sum_{j=1}^{m} \lambda_j^V = 0
$$

### 交叉验证准则:
$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{U}_{-i}(s_i) - U(s_i))^2}
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| cross_variogram_model | str | 'exponential' | 交叉变异函数模型 |
| n_neighbors | int | 10 | 近邻数量 |

## 数据规格

### 输入
| 数据 | 格式 | 说明 |
|-----|------|------|
| 主变量观测 | array | 目标变量（如PM2.5） |
| 辅助变量 | array | CMAQ或其他相关变量 |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 共克里金预测 | array | μg/m³ |

## 方法指纹
MD5: cokriging_multivariate_method

## 实现检查清单
- [ ] 核心公式已验证
- [ ] 共克里金求解器已实现
