# 复现指令: FCopt 优化加权融合法

## 方法信息
- **方法名**: FCopt
- **文献**: Friberg et al., ES&T 2016
- **核心公式**:
  - W = [R1 - R2] * R1 / ([R1 - R2]^2 + R1*(1-R1)*(1-Wmin))
  - FCopt = W * FC1 + (1-W) * FC2
- **实现文件**: CodeWorkSpace/复现方法代码/FCopt.py

## 算法步骤

### 1. FC1组件
- FC1 = krig(OBS/CMAQ) * FC_annual
- 与FC1Kriging方法相同

### 2. FC2组件
- FC2 = CTM_adj * beta_season
- 与FC2ScaleCMAQ方法相同

### 3. 计算时空相关性R1
- R1(s,t) = R_coll + (1 - R_coll) * exp(-x/r)
- x = 到最近观测的距离

### 4. 计算CMAQ时间相关性R2
- R2 = corr(OBS, CMAQ) 在监测站点

### 5. 计算优化权重W
- W = [R1 - R2] * R1 / ([R1 - R2]^2 + R1*(1-R1)*(1-Wmin))

### 6. 最终融合
- FCopt = W * FC1 + (1-W) * FC2

## 关键参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| W_min | 0.0 | 最小权重 |
| variogram_model | 'exponential' | 半变异函数模型 |
| k | 20 | 近邻数量 |

## 验证要求
- 十折交叉验证
- 指标: R², MAE, RMSE, MB

## 代码调用
```python
from FCopt import FCoptFusion

method = FCoptFusion(W_min=0.0, variogram_model='exponential', k=20)
method.fit(X_obs, y_obs, y_model_obs)
y_pred = method.predict(X_grid, y_model_grid)
```
