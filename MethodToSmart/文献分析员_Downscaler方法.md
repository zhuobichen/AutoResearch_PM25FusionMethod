# 【可执行方法规范】

## 方法名称
Downscaler - Kriging残差插值法

## 文献来源
- 论文标题：《PM2.5 CMAQ融合方法研究背景》
- 来源：背景文档
- 关键章节：3.2.5

## 核心公式
$$
P_{Downscaler}(s_0) = M(s_0) + Kriging(O(s_i) - M(s_i))
$$

## 参数清单
| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| numit | int | 2500 | MCMC迭代次数 |
| burn | int | 500 | 燃烧期 |
| thin | int | 1 |  thinning |
| neighbor | int | 3 | 近邻数量 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 网格坐标 | array | (n_grid, 2) | 度 |
| 监测站点坐标 | array | (n_obs, 2) | 度 |
| CMAQ网格值 | array | (n_grid,) | μg/m³ |
| 监测站点观测值 | array | (n_obs,) | μg/m³ |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 融合网格 | array | μg/m³ |

## 随机性
- [ ] 是（MCMC采样，带随机种子）

## 方法指纹
MD5: downscaler_method_fingerprint

## 实现检查清单
- [x] 核心公式已验证
- [x] 边界条件已处理
- [x] 单元测试通过
