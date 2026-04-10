# 复现指令: Gen-Friberg 广义CMAQ观测融合法

## 方法信息
- **方法名**: Gen-Friberg
- **文献**: Li et al., Environmental Modelling and Software 2025
- **核心公式**:
  - FC1: krig(OBS/CMAQ) * FC_annual
  - FC2: CTM_adj * beta_season
  - FC_final = W * FC1 + (1-W) * FC2
- **实现文件**: CodeWorkSpace/复现方法代码/GenFriberg.py

## 算法步骤

### 1. 年均值校正回归
- OBS = alpha + beta * CMAQ
- 计算年均场 FC_annual = alpha + beta * CMAQ

### 2. FC1融合
- R = OBS / mean(OBS)
- krig_R = 克里金插值(R)
- FC1 = krig_R * FC_annual

### 3. FC2融合
- CTM_adj = CMAQ * (FC_annual / CMAQ_annual)
- beta_season = 季节校正因子
- FC2 = CTM_adj * beta_season

### 4. 优化融合
- 计算R1(时空相关性) 和 R2(CMAQ时间相关性)
- W = R1 * (1-R2) / (R1*(1-R2) + R2*(1-R1))
- FC_final = W * FC1 + (1-W) * FC2

## 关键参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| regression_mode | 'auto' | 'linear', 'exponential', 'auto' |
| variogram_model | 'exponential' | 半变异函数模型 |
| n_folds | 10 | 交叉验证折数 |
| k | 20 | 近邻数量 |

## 验证要求
- 十折交叉验证
- 指标: R², MAE, RMSE, MB

## 代码调用
```python
from GenFriberg import GenFribergFusion

method = GenFribergFusion(regression_mode='auto', variogram_model='exponential', k=20)
method.fit(X_obs, y_obs, y_model_obs)
y_pred = method.predict(X_grid, y_model_grid)
```
