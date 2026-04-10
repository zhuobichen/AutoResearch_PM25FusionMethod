# 【可执行方法规范】

## 方法名称
Stacking-Ensemble - 堆叠集成学习方法 (Stacking Ensemble for CTM-Observation Fusion)

## 文献来源
- 论文标题: (参考 Lyu et al. 相关研究)
- 方法: 将随机森林、神经网络、梯度提升机(GLM)的预测进行堆叠集成
- 融合CMAQ、气象数据、土地利用数据

## 核心公式

### 基础模型预测:
$$
\hat{y}_{RF} = RF(X)
$$
$$
\hat{y}_{NN} = NN(X)
$$
$$
\hat{y}_{GBM} = GBM(X)
$$

### 元学习器（线性组合）:
$$
\hat{y}_{final} = w_1 \cdot \hat{y}_{RF} + w_2 \cdot \hat{y}_{NN} + w_3 \cdot \hat{y}_{GBM}
$$
其中 $w_i \geq 0$, $\sum w_i = 1$

### 带残差校正的最终输出:
$$
\hat{y}_{final} = \hat{y}_{stacking} + krig(y - \hat{y}_{stacking})
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| base_models | list | [RF, NN, GBM] | 基础模型列表 |
| meta_model | str | 'ridge' | 元学习器类型 |
| cv_folds | int | 5 | 交叉验证折数 |

## 数据规格

### 输入
| 数据 | 格式 | 说明 |
|-----|------|------|
| CMAQ模拟 | array | 网格浓度场 |
| 气象数据 | array | 温度、湿度、风等 |
| 土地利用 | array | 土地覆盖类型 |
| 坐标 | array | 经纬度 |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 融合预测 | array | μg/m³ |

## 随机性
- [ ] 是（机器学习训练带随机初始化）

## 方法指纹
MD5: stacking_ensemble_fusion

## 实现检查清单
- [ ] 基础模型已实现
- [ ] 元学习器已实现
- [ ] 克里金残差校正已实现
