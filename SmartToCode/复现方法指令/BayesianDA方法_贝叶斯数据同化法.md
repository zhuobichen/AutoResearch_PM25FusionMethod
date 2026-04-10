# 复现指令: Bayesian-DA 贝叶斯数据同化法

## 方法信息
- **方法名**: Bayesian-DA
- **文献**: Chianese et al., Ecological Modelling 2018
- **核心公式**: J(b) = (F(x-b) - y)^T P^{-1}(F(x-b) - y) + (b - b_0)^T Q^{-1}(b - b_0)
- **实现文件**: CodeWorkSpace/复现方法代码/BayesianDA.py

## 算法步骤

### 1. 数据准备
- 输入: X_obs(站点坐标), y_obs(监测值), y_model_obs(CMAQ预测值)
- 计算初始偏差: bias = y_obs - y_model_obs

### 2. 构建曲率矩阵D
- 使用站点坐标构建图的拉普拉斯算子
- 用于平滑约束项: epsilon * (Db)^T(Db)

### 3. 变分贝叶斯迭代
- E步: 给定超参数，估计偏差场b
- M步: 更新超参数 omega, epsilon, delta
- 成本函数: J = omega * ||bias||^2 + epsilon * b^T D^T D b + delta * ||b||^2

### 4. 网格预测
- 使用高斯权重IDW插值偏差到网格点
- 融合: y_pred = y_model_grid + bias_kriged

## 关键参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| omega | 1.0 | 观测拟合权重 |
| epsilon | 1e-2 | 平滑正则化 |
| delta | 1e-2 | 小偏差权重 |
| max_iter | 100 | 最大迭代 |

## 验证要求
- 十折交叉验证
- 指标: R², MAE, RMSE, MB
- 网格预测+站点提取验证

## 代码调用
```python
from BayesianDA import BayesianDA

method = BayesianDA(omega=1.0, epsilon=1e-2, delta=1e-2)
method.fit(X_obs, y_obs, y_model_obs)
y_pred = method.predict(X_grid, y_model_grid)
```
