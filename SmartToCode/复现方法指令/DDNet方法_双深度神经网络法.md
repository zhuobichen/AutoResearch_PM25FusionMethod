# 复现方法指令

## 方法名称
DDNet双深度神经网络实时预报与数据同化法 (Dual Deep Neural Networks for PM2.5 Forecasting and Data Assimilation)

## 输入数据
- 监测站坐标：shape (n, 2)
- CMAQ网格数据：shape (lat, lon, time)
- 气象变量：shape (lat, lon, time, n_met)
- 历史PM2.5序列：shape (n, T)
- AOD550卫星数据（可选）：shape (lat, lon, time)

## 输出数据
- 融合结果：shape (lat, lon, time)
- 分析场（经过DANet校正）：shape (lat, lon, time)
- 预报场（PredNet输出）：shape (lat, lon, time)

## 核心公式

### PredNet预测网络
$$
\hat{Y}_{t+\Delta t} = PredNet(Y_{t}, Y_{t-\Delta t}, ...; \theta_p)
$$
输入：历史PM2.5序列、气象变量、地形
输出：t+Δt时刻的PM2.5预报场

### DANet数据同化网络
$$
Y_{t}^{analysis} = DANet(\hat{Y}_{t}, obs_{t}; \theta_d)
$$
DANet学习PredNet预报与监测站观测之间的系统性偏差，并输出校正后的分析场

### 迭代预报循环
$$
Y_{t}^{analysis} \xrightarrow{PredNet} \hat{Y}_{t+\Delta t}^{forecast} \xrightarrow{DANet} Y_{t+\Delta t}^{analysis}
$$

### 损失函数（MAE）
$$
L_{Pred} = \sum_{grid} |Y_{pred}(s,t) - Y_{true}(s,t)|
$$
$$
L_{DAN} = \sum_{station} |Y_{analysis}(s_i,t) - Y_{obs}(s_i,t)|
$$

## 关键步骤
1. **PredNet训练**：在CMAQ重分析数据上训练多变量预报网络，学习气象-浓度条件转移
2. **DANet训练**：在学习预报-观测偏差的监测站上训练，输入为预报场和观测场，输出偏差校正量
3. **迭代同化**：在每个同化时刻，用DANet校正PredNet的预报，得到分析场
4. **网格插值**：将监测站偏差校正量通过IDW/Kriging插值到网格点
5. **滚动预报**：分析场作为下一步PredNet输入，循环推进

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| T_history | int | 72 | 输入历史时间步(hours) |
| delta_t | int | 3 | 预报时间步(hours) |
| DA_interval | int | 12 | 同化间隔(hours) |
| hidden_dim | int | 128 | 隐变量维度 |
| n_stations | int | - | 监测站数量 |
| learning_rate | float | 1e-3 | 学习率 |
| batch_size | int | 32 | 批大小 |

## 与系统的适配

本方法将CMAQ视为"预报"，监测站观测视为"真值"，通过DANet学习两者之间的偏差场，实现：
- CMAQ系统性偏差的网格级校正
- 监测数据的空间插值（通过偏差场的IDW/Kriging）
- 迭代推进实现时间连续的融合分析

## 创新判定
- 是否使用权重学习（Ridge/Lasso/线性回归）？否
- 是否有物理可解释性？PredNet为数据驱动，DANet为偏差学习，均有一定可解释性
- 是否保留：是（单模型双网络，无加权集成）

## 方法指纹
MD5: `ddnet_v1_prednet_danet_dual_system`

## 复现来源
- 文献分析员_DDNet双深度网络PM25预报法_20260409.md

## 随机性
- [x] 是（神经网络训练随机性）

## 验证方法
- 十折CV计算R²和RMSE
- 对比CMAQ原始输出的改善
