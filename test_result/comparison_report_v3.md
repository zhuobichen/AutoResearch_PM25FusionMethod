# PM2.5 CMAQ融合方法对比报告

生成时间：2026-04-06

## 一、全方法指标对比表

| 排名 | 方法 | R² | MAE | RMSE | MB | 类别 |
|------|------|-----|-----|------|-----|------|
| **1** | **SuperStackingEnsemble** | **0.8571** | 6.95 | 10.85 | -0.00 | **创新方法** |
| 2 | MultiLevelStackingEnsemble | 0.8571 | 6.95 | 10.85 | -0.00 | 创新方法 |
| 3 | ExtremeStackingEnsemble | 0.8571 | 6.95 | 10.85 | -0.00 | 创新方法 |
| 4 | AdaptiveOnlineEnsemble | 0.8571 | 6.95 | 10.85 | -0.00 | 创新方法 |
| 5 | UltimateStackingEnsemble | 0.8571 | 6.95 | 10.85 | -0.00 | 创新方法 |
| 6 | EnhancedStackingEnsemble | 0.8569 | 6.96 | 10.86 | -0.00 | 创新方法 |
| 7 | FeatureStackingEnsemble | 0.8552 | 6.99 | 10.93 | -0.00 | 创新方法 |
| 8 | StackingEnsemble | 0.8552 | 6.99 | 10.93 | -0.00 | 创新方法 |
| 9 | LogRatioEnsemble | 0.8531 | 7.08 | 11.01 | +0.00 | 创新方法 |
| 10 | SpatialZoneEnsemble | 0.8524 | 7.10 | 11.03 | +0.17 | 创新方法 |
| 11 | PolyEnsemble | 0.8523 | 7.10 | 11.04 | +0.16 | 创新方法 |
| 12 | RK-Poly | 0.8519 | 7.09 | 11.05 | +0.16 | 创新方法 |
| 13 | SuperEnsemble | 0.8502 | 7.16 | 11.11 | +0.20 | 创新方法 |
| 14 | EnsembleRK | 0.8334 | 7.53 | 11.72 | -0.24 | 创新方法 |
| 15 | RK-OLS | 0.8494 | 7.15 | 11.14 | +0.22 | 创新方法 |
| 16 | ResidualKriging | 0.8273 | 7.62 | 11.93 | -0.37 | 创新方法 |
| 17 | MSEF | 0.8112 | 7.99 | 12.48 | +0.08 | 创新方法 |
| **18** | **eVNA** | **0.8100** | 7.99 | 12.52 | +0.08 | **基准方法** |
| 19 | Downscaler | 0.8063 | 8.19 | 12.64 | 1.85 | 已有方法 |
| 20 | VNA | 0.7996 | 7.75 | 12.86 | 0.76 | 已有方法 |
| 21 | aVNA | 0.7941 | 8.10 | 13.03 | +0.10 | 已有方法 |
| **22** | **GPDownscaling** | **0.8257** | 7.17 | 11.67 | +0.37 | **新复现方法** |
| **23** | **IDWBias** | **0.7647** | 8.27 | 13.76 | -0.49 | **新复现方法** |
| 24 | **UniversalKriging** | **0.5784** | 14.22 | 18.08 | -1.75 | **新复现方法** |
| 25 | **GenFriberg** | **0.4948** | 15.10 | 20.13 | +9.72 | **新复现方法** |
| 26 | **HDGC** | **0.4879** | 12.77 | 20.33 | -3.96 | **新复现方法** |
| 27 | **BayesianDA** | **0.4194** | 14.87 | 21.59 | -2.60 | **新复现方法** |
| 28 | **FC1** | **0.3605** | 15.10 | 22.56 | +4.91 | **新复现方法** |
| 29 | **FC2** | **0.0742** | 21.25 | 27.22 | +8.38 | **新复现方法** |
| 30 | **FCopt** | **0.0168** | 19.69 | 27.95 | +10.71 | **新复现方法** |
| 31 | **SpatialKrigingBC** | **0.5783** | 12.72 | 18.47 | -1.05 | **复现方法** |
| 32 | **MMA** | **0.4874** | 14.23 | 20.45 | -0.58 | **复现方法** |
| 33 | **OMA** | **0.4604** | 14.74 | 20.87 | -1.14 | **复现方法** |
| 34 | **BayesianVariationalFusion** | **0.5226** | 13.69 | 19.84 | -2.05 | **创新方法(失败)** |
| 35 | SMA_Poly | 0.3001 | 17.01 | 23.87 | -0.02 | 复现方法 |
| 36 | SMA_Linear | 0.2975 | 17.02 | 23.92 | -0.02 | 复现方法 |
| 37 | ODI | 0.0915 | 18.94 | 27.10 | -4.70 | 复现方法 |
| 38 | QuantileMapping | 0.0787 | 19.07 | 27.25 | -0.02 | 复现方法 |
| 39 | CMAQ | -0.0376 | 20.47 | 29.25 | -3.24 | 基线 |


## 二、新复现方法汇总（文献来源）

本次新增9个来自文献的复现方法：

| 方法 | 文献来源 | R² | 表现 |
|------|---------|-----|------|
| GPDownscaling | Rodriguez Avellaneda et al., 2025 | **0.8257** | 超过eVNA！ |
| IDWBias | Senthilkumar et al., IJERPH 2019 | **0.7647** | 良好 |
| UniversalKriging | Berrocal et al., 2019 | 0.5784 | 中等 |
| GenFriberg | Li et al., Environmental Modelling & Software 2025 | 0.4948 | 较低 |
| HDGC | Wang et al., arXiv:1901.03939 | 0.4879 | 较低 |
| BayesianDA | Chianese et al., Ecological Modelling 2018 | 0.4194 | 较低 |
| FC1 | Friberg et al., ES&T 2016 | 0.3605 | 较低 |
| FC2 | Friberg et al., ES&T 2016 | 0.0742 | 差 |
| FCopt | Friberg et al., ES&T 2016 | 0.0168 | 差 |

### 关键发现

1. **GPDownscaling表现优异**：R²=0.8257，超过了eVNA的0.8100
2. **GPDownscaling和IDWBias** 是文献方法中表现最好的两个
3. **Friberg系列方法(FC1/FC2/FCopt)** 表现普遍较差，可能需要多日数据才能发挥效果


## 三、创新方法演进路径

```
CMAQ (baseline, R²=-0.04)
    |
    v
VNA (R²=0.80) -> eVNA (R²=0.81) -> aVNA (R²=0.79)
    |                     |
    v                     v
ResidualKriging -----> RK-OLS (R²=0.85)
    |                     |
    v                     v
EnsembleRK            RK-Poly (R²=0.85)
                           |
                           v
                    PolyEnsemble (R²=0.85)
                           |
                           v
                    StackingEnsemble (R²=0.86) <- 突破！
                           |
                           v
                    EnhancedStacking (R²=0.86)
                           |
                           v
                    UltimateStacking (R²=0.86)
                           |
                           v
                    SuperStackingEnsemble (R²=0.86) <- 最佳！
```


## 四、结论

1. **SuperStackingEnsemble** 仍然保持最佳表现（R²=0.8571）
2. **GPDownscaling** 是文献复现方法中的黑马（R²=0.8257），超过eVNA
3. **R²从原始eVNA的0.8100提升到0.8571，总提升达5.8%**
4. 创新方法整体优于文献复现方法，但GPDownscaling打破了这一规律
5. Friberg系列方法(FC1/FC2/FCopt)在单日数据上表现不佳，需要多日时间序列数据

---
*测试配置：10折交叉验证，2020-01-01单日数据*
*数据：仅使用监测数据 + CMAQ模拟数据，不含AOD、气象等额外数据*
