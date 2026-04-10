# 方法文档清单 (INVENTORY)

生成时间: 2026-04-09

## 已有方法文档

| 序号 | 文件名 | 方法类型 |
|-----|--------|---------|
| 1 | 文献分析员_AQNet时空神经网络法.md | 神经网络 |
| 2 | 文献分析员_BayesianSpaceTimeKriging法.md | 贝叶斯克里金 |
| 3 | 文献分析员_CRNNSpatiotemporalPM25法.md | CNN+RNN |
| 4 | 文献分析员_CleanAir深度学习CMAQ替代法.md | 深度学习替代CMAQ |
| 5 | 文献分析员_Cokriging共克里金法.md | 共克里金 |
| 6 | 文献分析员_Downscaler方法.md | 降尺度 |
| 7 | 文献分析员_FC1克里金插值法.md | 克里金插值 |
| 8 | 文献分析员_FC2尺度CMAQ法.md | CMAQ尺度 |
| 9 | 文献分析员_FCopt优化融合法.md | 优化融合 |
| 10 | 文献分析员_GP降尺度法.md | 高斯过程降尺度 |
| 11 | 文献分析员_GWR地理加权回归法.md | 地理加权回归 |
| 12 | 文献分析员_GenFriberg广义融合法.md | 广义融合 |
| 13 | 文献分析员_HDGC监测偏差检测法.md | 偏差检测 |
| 14 | 文献分析员_IDW偏差加权融合法.md | IDW加权融合 |
| 15 | 文献分析员_KNNSINDy缺失数据填补法.md | 缺失填补 |
| 16 | 文献分析员_LUR土地使用回归法.md | 土地使用回归 |
| 17 | 文献分析员_MLE最优插值法.md | 最优插值 |
| 18 | 文献分析员_RF残差克里金校正法.md | 残差校正 |
| 19 | 文献分析员_Stacking集成学习方法.md | 集成学习 |
| 20 | 文献分析员_VNA方法.md | VNA |
| 21 | 文献分析员_aVNA方法.md | aVNA |
| 22 | 文献分析员_eVNA方法.md | eVNA |
| 23 | 文献分析员_论文分析方法总结.md | 总结 |
| 24 | 文献分析员_贝叶斯数据同化法.md | 贝叶斯同化 |
| 25 | 文献分析员_通用克里金PM25映射法.md | 通用克里金 |
| 26 | 文献分析员_论文分析报告.md | 报告 |

## 新增方法文档 (2026-04-09)

| 序号 | 文件名 | 方法类型 |
|-----|--------|---------|
| 27 | 文献分析员_TopoFlow地形感知神经网络法_20260409.md | 物理引导Transformer |
| 28 | 文献分析员_Zeeman深度学习化学传输模型法_20260409.md | 深度学习CTM |
| 29 | 文献分析员_GenDA生成式数据同化法_20260409.md | 扩散模型同化 |
| 30 | 文献分析员_EnsAI大气化学集合生成法_20260409.md | U-Net集合生成 |
| 31 | 文献分析员_AirFusion扩散概率空气质量预报法_20260409.md | 扩散概率预报 |
| 32 | 文献分析员_NeuroDDAF神经动态扩散平流场法_20260409.md | 物理引导神经ODE |
| 33 | 文献分析员_DDNet双深度网络PM25预报法_20260409.md | 双网络同化 |

## 待分析论文清单

从paper_list.json中识别的PM2.5/CMAQ相关论文(已处理):

1. Physics-Guided Inductive Spatiotemporal Kriging (SPIN) - 2025
2. Bayesian Multisource Fusion Model - 2025
3. China Regional 3km Downscaling (CorrDiff) - 2025
4. TopoFlow (Physics-guided Neural Networks) - 2026 ✓
5. GenDA (Generative Data Assimilation) - 2026 ✓
6. Zeeman (Deep Learning CTM) - 2025 ✓
7. EnsAI (Atmospheric Chemical Ensemble Emulator) - 2025 ✓
8. D-DNet (Dual Deep Neural Networks) - 2024 ✓
9. AirFusion (Diffusion-based Probabilistic) - 2026 ✓
10. NeuroDDAF (Neural Dynamic Diffusion-Advection) - 2026 ✓

## 方法分类汇总

| 类别 | 方法数 | 方法名 |
|-----|-------|--------|
| 深度学习类 | 12 | AQNet, CRNN, CleanAir, Deep-AIR, CNN-LSTM, D-DNet, NeuroDDAF, AirFusion等 |
| 克里金/空间插值类 | 6 | BayesianSpaceTimeKriging, Cokriging, FC1, FC2, IDW, RF残差校正 |
| 物理引导类 | 5 | TopoFlow, Zeeman, GenDA, NeuroDDAF, AirFusion |
| 集合/不确定性类 | 3 | EnsAI, AirFusion, GenDA |
| 回归类 | 4 | LUR, GWR, Downscaler, Stacking |
| 其他 | 3 | VNA, aVNA, eVNA, KNNSINDy, MLE, HDGC |
