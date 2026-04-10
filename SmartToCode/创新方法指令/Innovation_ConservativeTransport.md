# 创新方法指令

## 方法名称
ConservativeTransport - 质量守恒传输映射法 (Mass-Conservative Advection-Diffusion Mapping)

## 输入数据
- 监测站坐标：shape (n, 2)
- CMAQ网格数据：shape (lat, lon, time)
- 气象数据（u/v风速、边界层高度）：shape (lat, lon, time, n_met)
- 监测PM2.5：shape (n, time)

## 输出数据
- 融合结果：shape (lat, lon, time)

## 核心公式

### 质量守恒约束（核心创新）
$$
\int_{\Omega} c_{fusion}(s,t) ds = \int_{\Omega} CMAQ(s,t) ds
$$
对任意子区域$\Omega$，融合场保持CMAQ的总质量。

### 半拉格朗日平流（保证质量守恒）
$$
c_{adv}(s,t+\Delta t) = c(s_i, t) \quad \text{where } s = s_i + v(s_i, t)\Delta t
$$
追踪上游点$s_i$，质量跟随流线传输。

### 扩散修正（高斯修正项）
$$
c_{final}(s,t) = c_{adv}(s,t) + \sum_{i} w_i(s) \cdot (O_i - CMAQ_i)
$$
其中$w_i(s)$为基于距离的插值权重（与VNA相同，无学习），修正扩散导致的数值耗散。

### 传输映射算子
$$
T_{adv}: CMAQ \rightarrow c_{adv}
$$
$$
T_{diff}: \{O_i - CMAQ_i\} \rightarrow \Delta c_{diff}
$$
$$
c_{fusion} = T_{adv}(CMAQ) + T_{diff}(residual)
$$

### 局地质量平衡（受体导向）
$$
O(s_i) = \int_\Omega K(s_i, s) \cdot CMAQ(s) ds + \epsilon
$$
其中$K(s_i, s)$为源-受体矩阵，由风场和扩散率决定。

## 关键步骤
1. **构建传输算子**：基于气象场（u,v风场，PBLH）构建半拉格朗日传输映射
2. **平流传输**：CMAQ浓度沿风场方向传输，保持总质量
3. **扩散修正计算**：在监测站位置计算$CMAQ_i^{trans} - O_i$残差
4. **残差空间插值**：用对数-距离权重（与VNA相同，无学习）将残差插值到网格
5. **融合叠加**：$c_{fusion} = c_{adv} + \Delta c_{diff}$
6. **质量守恒检验**：验证融合场与CMAQ总质量偏差 < 0.1%

## 【创新点】
1. **质量守恒保证**：半拉格朗日传输保持CMAQ总质量不变，物理一致性与现有方法不同
2. **受体导向诊断**：源-受体矩阵$K$由气象场计算，可解释污染传输路径
3. **无权重学习**：残差插值使用固定的对数-距离权重，无任何学习参数
4. **物理可解释**：扩散修正项直接对应大气扩散耗散，可验证是否符合物理量级

## 创新判定
- 是否使用权重学习（Ridge/Lasso/线性回归）？否（固定对数-距离权重）
- 是否有物理可解释性？是（半拉格朗日平流+扩散修正，质量守恒约束）
- 是否保留：创新成立

## 创新优势
- 相比ResidualKriging：无残差建模，直接传输CMAQ结构
- 相比PhysicICNN-PDE：计算简单，无需神经网络训练
- 相比SuperStackingEnsemble：质量守恒约束，无加权集成
- 预期R²提升 >= 0.01（质量守恒约束减少非物理震荡）

## 风险假设
- 气象场质量影响传输算子精度
- 对于风场静稳情况（< 1m/s），平流传输退化
- 验证计划：检验融合场与CMAQ总质量偏差

## 方法指纹
MD5: `conservative_transport_mass_balance_v1`

## 输入输出格式
- 输入：监测CSV + CMAQ netCDF + 气象netCDF
- 输出：融合网格PM2.5 + 质量守恒检验报告
- 支持十折交叉验证
