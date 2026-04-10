# 复现指令: HDGC 监测偏差检测法

## 方法信息
- **方法名**: HDGC
- **文献**: Wang et al., 2019 (arXiv:1901.03939)
- **核心公式**: Z(s,t) = mu(s) + U(s,t), Y_i(t) = gamma_i * Z(s_i,t) + epsilon_i(t)
- **实现文件**: CodeWorkSpace/复现方法代码/HDGC.py

## 算法步骤

### 1. 数据准备
- 输入: X_obs(站点坐标), y_obs(监测值), y_model_obs(CMAQ预测值)

### 2. EM算法估计
- 初始化校准参数 gamma_i = 1.0
- E步: 估计隐藏场Z
- M步: 更新gamma_i

### 3. 偏差检测
- 计算每个站点的gamma标准误
- 标记准则: |gamma_i - 1| > 2 * SE(gamma_i) 为有偏站点

### 4. 网格预测
- 使用高斯权重IDW插值gamma到网格点
- 校正: y_pred = gamma_kriged * CMAQ_grid

## 关键参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_iter | 200 | EM最大迭代 |
| tol | 1e-4 | 收敛容忍度 |
| rho_s | None | 空间相关尺度 |
| k | 10 | 近邻数量 |

## 验证要求
- 十折交叉验证
- 指标: R², MAE, RMSE, MB

## 代码调用
```python
from HDGC import HDGC

method = HDGC(max_iter=200, tol=1e-4, k=10)
method.fit(X_obs, y_obs, y_model_obs)
y_pred = method.predict(X_grid, y_model_grid)
```
