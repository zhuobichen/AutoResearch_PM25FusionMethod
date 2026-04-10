# 复现指令: FC2 尺度CMAQ融合法

## 方法信息
- **方法名**: FC2
- **文献**: Friberg et al., ES&T 2016
- **核心公式**: FC2(s,t) = CMAQ(s,t) * beta * FC_annual(s) / CMAQ_annual(s) * beta_season(t)
- **实现文件**: CodeWorkSpace/复现方法代码/FC2.py

## 算法步骤

### 1. 年度回归
- OBS = alpha + beta * CMAQ
- 计算校正因子: ratio = FC_annual / CMAQ_annual

### 2. 季节校正
- beta_season(t) = 1 + A * cos[2*pi*(t - t_max)/365.25]
- 简化: 使用常数 seasonal_factor (因为只有单日数据)

### 3. 网格预测
- correction = IDW(ratio) * seasonal_factor
- FC2 = CMAQ_grid * correction

## 特点
- 不依赖空间稀疏的观测网络
- 误差与到观测的距离无关
- 主要受CMAQ模拟准确性限制

## 关键参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| seasonal_correction | True | 是否应用季节校正 |
| k | 20 | 近邻数量 |

## 验证要求
- 十折交叉验证
- 指标: R², MAE, RMSE, MB

## 代码调用
```python
from FC2 import FC2ScaleCMAQ

method = FC2ScaleCMAQ(seasonal_correction=True, k=20)
method.fit(X_obs, y_obs, y_model_obs)
y_pred = method.predict(X_grid, y_model_grid)
```
