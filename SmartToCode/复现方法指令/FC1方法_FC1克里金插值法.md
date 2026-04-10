# 复现指令: FC1 克里金插值融合法

## 方法信息
- **方法名**: FC1
- **文献**: Friberg et al., ES&T 2016
- **核心公式**: FC1(s,t) = krig(OBS(t)/OBS) * FC_annual(s)
- **实现文件**: CodeWorkSpace/复现方法代码/FC1.py

## 算法步骤

### 1. 归一化观测
- R = OBS / mean(OBS)

### 2. CMAQ年场校正
- OBS_mean = alpha + beta * CMAQ_mean
- FC_annual = alpha + beta * CMAQ

### 3. 半变异函数拟合
- 使用归一化比值的残差拟合半变异函数
- 模型: gamma(h) = c0 + c*(1 - exp(-h/a))

### 4. 克里金插值
- krig_R = 克里金插值(R) 到网格点

### 5. 反归一化
- FC1 = krig_R * FC_annual

## 关键参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| variogram_model | 'exponential' | 半变异函数模型 |
| k | 20 | 近邻数量 |

## 验证要求
- 十折交叉验证
- 指标: R², MAE, RMSE, MB

## 代码调用
```python
from FC1 import FC1Kriging

method = FC1Kriging(variogram_model='exponential', k=20)
method.fit(X_obs, y_obs, y_model_obs)
y_pred = method.predict(X_grid, y_model_grid)
```
