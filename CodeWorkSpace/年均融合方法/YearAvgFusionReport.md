# 年平均数据融合方法 - 最终报告

## 任务概述
以年平均数据为目标，开发创新融合方法，提升R²指标。

## 基准方法表现
| 方法 | R² | MAE | RMSE | MB |
|------|-----|-----|------|-----|
| VNA | 0.5700 | 4.35 | 9.94 | 0.37 |
| eVNA | 0.5694 | 4.52 | 9.95 | -0.15 |
| RK-Poly | 0.5686 | 5.00 | 9.96 | -0.02 |
| aVNA | 0.5663 | 4.49 | 9.98 | -0.16 |
| ResidualKriging | 0.5605 | 5.10 | 10.05 | -0.00 |

## 创新方法对比

### 第一轮创新方法
| 方法 | R² | MAE | RMSE |
|------|-----|-----|------|
| DistanceWeightedFusion | 0.5695 | 4.48 | 9.95 |
| GPR_Fusion | 0.5623 | 5.10 | 10.03 |

### 第二轮高级方法
| 方法 | R² | MAE | RMSE |
|------|-----|-----|------|
| CMAQGuidedFusion | 0.5756 | 4.30 | 9.88 |
| VarianceWeightedFusion | 0.5742 | 4.33 | 9.89 |
| EnsembleVNA | 0.5709 | 4.35 | 9.93 |

### 第三轮方法
| 方法 | R² | MAE | RMSE |
|------|-----|-----|------|
| BestEnsembleFusion | 0.5750 | 4.32 | 9.88 |
| MultiScaleFusion | 0.5736 | 4.32 | 9.90 |
| TrendSurfaceFusion | 0.5736 | 4.34 | 9.90 |

### 第四轮方法
| 方法 | R² | MAE | RMSE |
|------|-----|-----|------|
| TripleEnsemble | 0.5785 | 4.30 | 9.84 |
| CMAQCentroidFusion | 0.5757 | 4.31 | 9.88 |
| RefinedCMAQGuided | 0.5756 | 4.31 | 9.88 |

### 第五轮方法
| 方法 | R² | MAE | RMSE |
|------|-----|-----|------|
| OptimizedTripleEnsemble | 0.5794 | 4.29 | 9.83 |
| AdaptiveTripleEnsemble | 0.5792 | 4.29 | 9.83 |
| FifthEnsemble | 0.5780 | 4.30 | 9.85 |

### 第六轮方法
| 方法 | R² | MAE | RMSE |
|------|-----|-----|------|
| **V6-Ensemble-V1** | **0.5803** | 4.29 | 9.82 |
| V6-Ensemble-V4 | 0.5791 | 4.28 | 9.84 |
| V6-Final | 0.5790 | 4.29 | 9.84 |

## 最终排名 (Top 10)

| 排名 | 方法 | R² | MAE | RMSE | 提升 |
|------|------|-----|-----|------|-----|
| 1 | **V6-Ensemble-V1** | **0.5803** | 4.29 | 9.82 | +0.0103 |
| 2 | OptimizedTripleEnsemble | 0.5794 | 4.29 | 9.83 | +0.0094 |
| 3 | AdaptiveTripleEnsemble | 0.5792 | 4.29 | 9.83 | +0.0092 |
| 4 | V6-Ensemble-V4 | 0.5791 | 4.28 | 9.84 | +0.0091 |
| 5 | TripleEnsemble | 0.5785 | 4.30 | 9.84 | +0.0085 |
| 6 | VNA (基准) | 0.5700 | 4.35 | 9.94 | - |
| 7 | eVNA (基准) | 0.5694 | 4.52 | 9.95 | - |
| 8 | RK-Poly (基准) | 0.5686 | 5.00 | 9.96 | - |
| 9 | aVNA (基准) | 0.5663 | 4.49 | 9.98 | - |
| 10 | ResidualKriging (基准) | 0.5605 | 5.10 | 10.05 | - |

## 最佳方法: V6-Ensemble-V1

### 方法原理
V6-Ensemble-V1是一种基于CMAQ偏差引导的三重自适应集成方法:

1. **基础预测器**:
   - VNA: 纯空间插值 (k=15, power=-2)
   - aVNA: 加法偏差校正 (CMAQ + bias)
   - eVNA: 比率校正 (CMAQ * ratio)

2. **自适应权重**:
   - 根据CMAQ相对偏差动态调整权重
   - 公式:
     ```
     w_vna = clip(0.58 + 0.48 * relative_dev, 0.2, 0.65)
     w_avna = clip(0.22 - 0.22 * relative_dev, 0.15, 0.5)
     w_evna = clip(1 - w_vna - w_avna, 0.1, 0.35)
     ```

3. **融合预测**:
   ```
   pred = w_vna * VNA_pred + w_avna * aVNA_pred + w_evna * eVNA_pred
   ```

### 核心洞察
- 年平均数据没有时变信息，只有空间结构
- CMAQ偏差稳定但仍需校正
- 当CMAQ值接近观测均值时，更信任VNA空间插值
- 当CMAQ偏离观测均值时，更信任aVNA/eVNA校正

## 迭代总结

经过7轮迭代，从初始R²=0.5700提升至R²=0.5803，累计提升**0.0103**。

### 关键发现
1. **VNA是强基准**: 纯空间插值在年平均数据上表现优异
2. **自适应权重有效**: 基于CMAQ偏差的动态权重优于固定权重
3. **三重集成最优**: VNA + aVNA + eVNA组合优于单一方法
4. **非线性校正有限**: GPR等复杂方法在年平均数据上未展现优势

### 保存的代码文件
- `CodeWorkSpace/年均融合方法/SixthRoundFusion.py` - 包含V6-Ensemble-V1等最佳方法