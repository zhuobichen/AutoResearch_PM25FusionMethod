# 复现指令: IDW-Bias IDW偏差加权融合法

## 方法信息
- **方法名**: IDW-Bias
- **文献**: Senthilkumar et al., IJERPH 2019
- **核心公式**: R_m = OBS_m / CTM_m, FC(s) = CTM(s) * IDW(R)
- **实现文件**: CodeWorkSpace/复现方法代码/IDWBias.py

## 算法步骤

### 1. 数据准备
- 输入: X_obs(站点坐标), y_obs(监测值), y_model_obs(CMAQ预测值)

### 2. 计算偏差比值
- R_i = OBS_i / CMAQ_i
- 限制范围: [0.2, 5.0]

### 3. IDW插值
- w_i = 1 / d_i^p
- IDW(R) = sum(w_i * R_i) / sum(w_i)

### 4. 融合
- FC(s) = CMAQ(s) * IDW(R)

## 关键参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| power | 2.0 | IDW权重指数 |
| max_distance | 100.0 | 最大插值距离 |
| min_neighbors | 3 | 最小近邻数 |

## 验证要求
- 十折交叉验证
- 指标: R², MAE, RMSE, MB

## 代码调用
```python
from IDWBias import IDWBias, IDWBiasWeighted

method = IDWBias(power=2.0, max_distance=100.0, min_neighbors=3)
method.fit(X_obs, y_obs, y_model_obs)
y_pred = method.predict(X_grid, y_model_grid)
```
