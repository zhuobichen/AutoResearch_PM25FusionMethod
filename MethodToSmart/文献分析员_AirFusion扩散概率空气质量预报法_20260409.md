# 【可执行方法规范】

## 方法名称
空气融合扩散概率预报 (AirFusion: Diffusion-based Probabilistic Air Quality Forecasting)

## 文献来源
- 论文标题：Diffusion-based Probabilistic Air Quality Forecasting with Mechanistic Insight
- 作者/年份：Ding et al. / 2026年
- 关键章节：Section 2 / Section 4

## 核心公式

**AirFusion-S (空间插值)：**
$$
X_0(t_0) = \text{AirFusion-S}( obs_{t_0-\Delta t}, obs_{t_0} )
$$

**AirFusion-T (时间推进)：**
$$
X_0(t_0 + \Delta t) = \text{AirFusion-T}( X_0(t_0), X_0(t_0-\Delta t); met_{t_0}, met_{t_0+\Delta t} )
$$

**AirFusion-T-FT (微调版本)：**
$$
X_F(t_0 + \Delta t) = \text{AirFusion-T-FT}( X_0(t_0), X_0(t_0-\Delta t); met_{t_0}, met_{t_0+\Delta t} )
$$

**扩散模型去噪：**
$$
p_\theta(x^{(t-1)}|x^{(t)}) = \mathcal{N}(\mu_\theta(x^{(t)}, t), \Sigma_\theta(x^{(t)}, t))
$$

## 参数清单
| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| 空间分辨率 | float | 27km | 中国区域 |
| 时间分辨率 | int | 3h | 输出间隔 |
| 预报时效 | int | 6天 | 预报长度 |
| 集合成员数 | int | 30 | GEFS气象集合 |
| 微调数据 | int | ~5个月 | 观测数据量 |

## 数据规格
| 数据 | 格式 | 维度 |
|-----|------|-----|
| 地面观测 | CSV | 逐小时站点数据 |
| WRF-GC模拟 | netCDF | CTM输出 |
| 气象场 | GEFS集合 | 30成员 |
| 预报区域 | China | 72.8°-137.2°E, 16.4°-54.7°N |

## 实现步骤
1. **CTM预训练**：AirFusion-S和AirFusion-T在WRF-GC模拟数据上训练
2. **空间插值(AirFusion-S)**：将稀疏站点观测插值为连续2D污染场
3. **时间推进(AirFusion-T)**：给定当前和历史污染场+气象预报，预测下一时刻
4. **观测微调(AirFusion-T-FT)**：用最近5个月观测微调，修正CTM偏差
5. **集合预报**：用30个气象集合成员生成30个污染预报成员

## 方法指纹
MD5: `airfusion_v1_diffusion_ctm_hybrid`

## 随机性
- [x] 是  - [ ] 否（扩散模型本质随机）

## 备注
- 混合CTM+深度学习+观测校正
- 臭氧MDA8预报RMSE 26.9±5.7 μg/m³
- 比WRF-GC快4个数量级（40秒 vs 80CPU核心小时）
- 可快速适应排放变化（仅需1个月观测微调）
