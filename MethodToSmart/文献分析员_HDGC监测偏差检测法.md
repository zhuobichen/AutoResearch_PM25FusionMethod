# 【可执行方法规范】

## 方法名称
HDGC - 隐藏动态地统计校正法 (Hidden Dynamic Geostatistical Calibration)

## 文献来源
- 论文标题: "Bias detection of PM2.5 monitor readings using hidden dynamic geostatistical calibration model"
- 作者: Y. Wang, M. Xu, H. Huang, S. Chen
- 机构: 中国统计学年鉴 / 清华大学
- 日期: 2019
- arXiv: 1901.03939

## 核心公式

### 隐藏动态地统计模型 (HDGC):
模型假设PM2.5真值由一个隐藏的动态随机域表示:

$$
Z(s,t) = \mu(s) + U(s,t)
$$
其中:
- $\mu(s)$ = 空间趋势函数
- $U(s,t)$ = 具有可分离时空协方差结构的随机场

### 校准组件:
每个监测站i有一个校准参数 $\gamma_i$:
$$
Y_i(t) = \gamma_i \cdot Z(s_i, t) + \epsilon_i(t)
$$
其中 $Y_i(t)$ = 观测值，$\epsilon_i(t)$ = 观测误差

### 时空协方差结构:
$$
Cov(U(s_i, t), U(s_j, t')) = \sigma^2 \cdot \exp\left(-\frac{d_{ij}}{\rho_s} - \frac{|t-t'|}{\rho_t}\right)
$$
其中:
- $d_{ij}$ = 站点i和j之间的距离
- $\rho_s$ = 空间相关尺度
- $\rho_t$ = 时间相关尺度

### EM算法估计:
E步: 估计隐藏场 $U(s,t)$
M步: 更新参数 $\theta = {\gamma_i, \sigma^2, \rho_s, \rho_t}$

### 偏差检测准则:
如果 $|\gamma_i - 1| > 2 \cdot SE(\gamma_i)$，则站点i被标记为有偏

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| max_iter | int | 200 | EM算法最大迭代 |
| tol | float | 1e-4 | 收敛容忍度 |
| rho_s | float | data-fitted | 空间尺度参数 |
| rho_t | float | data-fitted | 时间尺度参数 |
| n_neighbors | int | 5 | 空间近邻数 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 监测数据 | array | (n_station, n_time) | μg/m³ |
| 站点坐标 | array | (n_station, 2) | 度 |
| 时间索引 | array | (n_time,) | 日期 |

### 输出
| 数据 | 格式 | 说明 |
|-----|------|------|
| 校准参数 | array | 每个站点的gamma值 |
| 偏差标记 | array | 布尔数组，标记有偏站点 |
| 隐藏场 | array | 估计的真值场 |

## 实现步骤

1. **初始化**: 设置参数初始值
2. **E步**: 给定当前参数，估计隐藏场 $U(s,t)$
3. **M步**: 最大化Q函数，更新所有参数
4. **迭代**: 重复E步和M步直到收敛
5. **偏差检测**: 识别校准参数显著偏离1的站点
6. **校正**: 使用估计的校准参数校正有偏读数

## 应用场景
- 检测PM2.5监测站的系统性偏差
- 校正仪器误差或维护问题导致的偏差读数
- 提高监测数据质量

## 随机性
- [ ] 是（EM算法可能收敛到局部最优）

## 方法指纹
MD5: hdgc_bias_detection_method

## 实现检查清单
- [x] 核心公式已验证
- [x] EM算法已实现
- [x] 偏差检测准则已实现
