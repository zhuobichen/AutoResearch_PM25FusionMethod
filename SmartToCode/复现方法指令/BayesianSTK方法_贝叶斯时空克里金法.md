# 复现方法指令

## 方法名称
Bayesian-STK - 贝叶斯时空克里金法 (Bayesian Spatiotemporal Kriging)

## 输入数据
- 监测站坐标：shape (n, 2)
- CMAQ网格数据：shape (lat, lon, time)
- 监测PM2.5：shape (n, time)
- 气象数据（可选）：shape (lat, lon, time, n_met)

## 输出数据
- 融合结果：shape (lat, lon, time)
- 不确定性场：shape (lat, lon, time)（后验标准差）

## 核心公式

### 时空随机场模型
$$
Y(s,t) = X(s,t)\beta + U(s,t) + \epsilon(s,t)
$$
其中：
- $X(s,t)$ = CMAQ网格作为时空协变量
- $\beta$ = 回归系数（斜率，校正CMAQ偏差）
- $U(s,t)$ = 时空随机场（均值0，协方差已知）
- $\epsilon(s,t)$ = 独立观测误差

### 时空协方差函数
$$
Cov(U(s_i,t_i), U(s_j,t_j)) = \sigma^2 \exp\left(-\frac{d_{ij}}{\rho_s} - \frac{|t_i-t_j|}{\rho_t}\right)
$$
其中：
- $d_{ij}$ = 站点间空间距离
- $\rho_s$ = 空间相关尺度
- $\rho_t$ = 时间相关尺度

### 贝叶斯推断（后验分布）
$$
p(\beta, \sigma^2, \rho_s, \rho_t | Y) \propto p(Y | \beta, \sigma^2, \rho_s, \rho_t) \cdot p(\beta) \cdot p(\sigma^2) \cdot p(\rho_s) \cdot p(\rho_t)
$$

### 后验预测分布
$$
p(Y_0(s_0,t_0) | Y_{obs}) = \int p(Y_0 | \theta) p(\theta | Y_{obs}) d\theta
$$

## 关键步骤
1. **构建时空协方差矩阵**：基于站点位置和时间差计算
2. **先验设定**：$\beta \sim N(0, \sigma_\beta^2)$，$\sigma^2 \sim IG(a,b)$，$\rho_s, \rho_t \sim U(l_{min}, l_{max})$
3. **MCMC采样**：使用Gibbs Sampling交替采样$(\beta, \sigma^2)$和$(\rho_s, \rho_t)$
4. **克里金预测**：对每个网格点计算后验均值和方差
5. **不确定性传播**：汇总MCMC样本得到预测不确定性

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| spatial_range | float | 50km | 空间相关尺度初始值 |
| temporal_range | float | 12h | 时间相关尺度初始值 |
| n_iter | int | 5000 | MCMC迭代次数 |
| burn_in | int | 2000 | 预烧期 |
| a_prior | float | 0.1 | 逆Gamma先验参数a |
| b_prior | float | 0.1 | 逆Gamma先验参数b |
| sigma_beta_prior | float | 10.0 | $\beta$先验标准差 |

## 与系统的适配

本方法将CMAQ作为固定协变量$X(s,t)$，通过贝叶斯推断：
- 校正CMAQ的系统性偏差（$\beta$斜率）
- 建模时空残差结构（$U(s,t)$）
- 提供完整的不确定性量化

## 创新判定
- 是否使用权重学习（Ridge/Lasso/线性回归）？否（贝叶斯推断，全局采样）
- 是否有物理可解释性？是（时空协方差函数有物理意义）
- 是否保留：保留（复现方案）

## 方法指纹
MD5: `bayesian_stk_spatiotemporal_kriging_mcmc`

## 复现来源
- 文献分析员_BayesianSpaceTimeKriging法.md

## 随机性
- [x] 是（MCMC采样固有随机性）

## 验证方法
- 十折CV计算R²和RMSE
- 对比CMAQ原始输出的改善
- 检验不确定性校准性（预测区间覆盖率）
