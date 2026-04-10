# 复现指令: GP-Downscaling 高斯过程降尺度法

## 方法信息
- **方法名**: GP-Downscaling
- **文献**: Rodriguez Avellaneda et al., 2025 (KAUST)
- **核心公式**: W_m(s) = sum_k a_mk * U_k(s) (多变量线性协同区域化模型)
- **实现文件**: CodeWorkSpace/复现方法代码/GPDownscaling.py

## 算法步骤

### 1. 数据准备
- 输入: X_obs(站点坐标), y_obs(监测值), y_model_obs(CMAQ预测值)

### 2. 尺度因子建模
- 建立线性关系: obs = alpha + beta * cmaq + epsilon
- 计算残差用于空间插值

### 3. 空间变异函数估计
- 计算实验半变异函数
- 使用Matérn协方差函数(简化版用指数协方差)
- 估计sill, range, nugget参数

### 4. 克里金插值
- 构建观测点协方差矩阵
- 对残差进行克里金插值

### 5. 网格预测
- 趋势项: alpha + beta * CMAQ_grid
- 残差插值: krig(residuals)
- 融合: y_pred = trend + residual_kriged

## 关键参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_latent | 3 | 潜在过程数量 |
| nu | 1.0 | Matérn平滑参数 |
| k | 20 | 近邻数量 |

## 验证要求
- 十折交叉验证
- 指标: R², MAE, RMSE, MB

## 代码调用
```python
from GPDownscaling import GPDownscaling

method = GPDownscaling(n_latent=3, nu=1.0, k=20)
method.fit(X_obs, y_obs, y_model_obs)
y_pred = method.predict(X_grid, y_model_grid)
```
