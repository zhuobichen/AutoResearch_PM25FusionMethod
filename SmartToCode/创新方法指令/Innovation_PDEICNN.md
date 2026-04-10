# 创新方法指令

## 方法名称
PhysicICNN-PDE：物理信息凸神经网络硬约束法 (Physics-Informed Convex Neural Network with Hard Advection-Diffusion Constraint)

## 输入数据
- 监测站坐标：shape (n, 2)
- CMAQ数据：shape (lat, lon, time)
- 气象数据（u/v风速、温度、边界层高度）：shape (lat, lon, time, n_met)
- 监测PM2.5：shape (n, time)

## 输出数据
- 融合结果：shape (lat, lon, time)

## 核心公式

### 硬PDE约束损失项（无权重学习）
$$
L_{PDE} = \frac{1}{N}\sum_{(s,t)}\left\| \frac{\partial c}{\partial t} - D\nabla^2 c + v\cdot\nabla c - S \right\|^2
$$
其中：
- $c$：神经网络预测的污染物浓度
- $D$：扩散系数（可学习标量或空间变量）
- $v=(u,v)$：风场（来自气象输入）
- $S$：源汇项（神经网络输出）

### 总损失函数
$$
L_{total} = L_{data} + \lambda \cdot L_{PDE}
$$
其中 $L_{data} = \sum_{station}|c_{pred}(s_i,t) - obs_i(t)|$（仅在监测站位置计算）

### ICNN凸性约束（防止非物理震荡）
$$
W_{enc} \geq 0, \quad W_{skip}^{(k)} \geq 0
$$
通过ReLU正权重化实现凸性，保证预测场单调平滑

### CMAQ偏差校正
$$
c_{final}(s,t) = CMAQ(s,t) + \Delta c_{PhysicICNN}(s,t)
$$
Δc通过PhysicICNN从(气象条件, CMAQ梯度)预测，替代残差克里金

## 关键步骤
1. **构建ICNN**：输入层=(CMAQ浓度, u, v, T, PBLH)，通过正权重凸层处理
2. **PDE约束注入**：在每个时间步计算浓度场的时空梯度，叠加为PDE损失项
3. **气象引导**：将u/v风场通过卷积层注入注意力，驱动平流项
4. **CMAQ条件化**：将CMAQ当前场作为残差预测的条件输入
5. **十折验证**：在监测站上进行交叉验证，调优λ

## 【创新点】
1. **无权重学习**：所有权重由神经网络优化，PDE项为物理约束而非数据驱动权重
2. **硬约束而非软约束**：PDE项直接作为损失函数项而非数据增强，避免过拟合
3. **物理可解释**：扩散系数D和源项S可直接解释为大气扩散率和排放源强度
4. **凸性保证**：ICNN结构防止非物理震荡

## 创新判定
- 是否使用权重学习（Ridge/Lasso/线性回归）？否
- 是否有物理可解释性？是（硬PDE约束，扩散/平流机制清晰）
- 是否保留：创新成立

## 创新优势
- 相比SuperStackingEnsemble：无加权集成，物理机制清晰，可迁移
- 相比普通ResidualKriging：PDE约束提供物理先验，适应性更强
- 预期R²提升 >= 0.02（CMAQ偏差有物理规律，PDE约束可捕捉）

## 风险假设
- PDE参数(D, S)可能过度拟合，需要充足数据
- 气象场分辨率需与CMAQ匹配
- 验证计划：十折CV对比ResidualKriging的RMSE

## 方法指纹
MD5: `physicicnn_pde_hard_constraint_v1`

## 输入输出格式
- 输入：监测CSV + CMAQ netCDF + 气象netCDF
- 输出：融合网格PM2.5
- 支持十折交叉验证
