# 复现指令: Universal-Kriging 通用克里金PM25映射法

## 方法信息
- **方法名**: Universal-Kriging
- **文献**: Berrocal et al., 2019 (arXiv:1904.08931)
- **核心公式**: Y(s) = X(s)*beta + U(s), Y_hat = X(s_0)*beta + sum_i lambda_i*(Y(s_i) - X(s_i)*beta)
- **实现文件**: CodeWorkSpace/复现方法代码/UniversalKriging.py

## 算法步骤

### 1. 数据准备
- 输入: X_obs(站点坐标), y_obs(监测值), y_model_obs(CMAQ预测值)

### 2. 趋势建模
- 使用CMAQ作为协变量拟合线性回归
- Y_trend = beta_0 + beta_1 * CMAQ

### 3. 计算残差
- R(s) = Y(s) - Y_trend(s)

### 4. 拟合半变异函数
- 使用残差拟合半变异函数参数
- 模型: gamma(h) = c0 + c*(1 - exp(-h/a))

### 5. 克里金预测
- 解通用克里金方程组
- Y_pred = trend + kriging(residuals)

## 关键参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| variogram_model | 'exponential' | 半变异函数模型 |
| drift | 'linear' | 趋势函数类型 |
| k | 20 | 近邻数量 |

## 验证要求
- 十折交叉验证
- 指标: R², MAE, RMSE, MB

## 代码调用
```python
from UniversalKriging import UniversalKriging

method = UniversalKriging(variogram_model='exponential', drift='linear', k=20)
method.fit(X_obs, y_obs, y_model_obs)
y_pred = method.predict(X_grid, y_model_grid)
```
