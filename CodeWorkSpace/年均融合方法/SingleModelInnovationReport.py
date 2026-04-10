"""
年平均数据融合 - 最终单模型创新报告
=====================================

项目目标:
- 基准: VNA R²=0.5700
- 目标: R² ≥ 0.5800 (提升 ≥ 0.01)
- 约束: 必须是单模型 (排除加权集成方法)

已排除的加权集成方法:
- V6-Ensemble-V1: R²=0.5803
- OptimizedTripleEnsemble: R²=0.5794
- AdaptiveTripleEnsemble: R²=0.5792
- TripleEnsemble: R²=0.5785
- 所有StackingEnsemble

=====================================
最终最佳单模型排名
=====================================

1. RefinedBayesianFusion: R²=0.5782, MAE=4.31, RMSE=9.85
   提升: +0.0082
   原理: 贝叶斯启发的权重融合，结合CMAQ偏差先验和似然

2. BayesianInspiredFusion (Round 11): R²=0.5781, MAE=4.31, RMSE=9.85
   提升: +0.0081
   原理: 贝叶斯权重结合先验(VNA)和似然(CMAQ校正)

3. CorrelationGuidedFusion (Round 11): R²=0.5773, MAE=4.30, RMSE=9.86
   提升: +0.0073
   原理: 根据CMAQ-O局部相关性调整融合权重

4. KernelAwareFusion (Round 10): R²=0.5770, MAE=4.32, RMSE=9.86
   提升: +0.0070
   原理: CMAQ感知的高斯核加权融合

5. TripleBandwidthFusion (Round 11): R²=0.5765, MAE=4.32, RMSE=9.87
   提升: +0.0065
   原理: 多尺度空间信息的融合

6. CMAQConditionedFusion (Round 8): R²=0.5759, MAE=4.30, RMSE=9.87
   提升: +0.0059
   原理: 根据CMAQ分位数条件选择融合策略

7. CMAQCentroidFusion (Round 3): R²=0.5757
   原理: CMAQ质心引导的融合

8. CMAQGuidedFusion (Round 2): R²=0.5756
   原理: CMAQ偏差引导的空间融合

9. SpatialLocalWeighting (Round 8): R²=0.5744
   提升: +0.0044
   原理: 空间邻域密度感知的加权融合

=====================================
结论
=====================================

最佳单模型: RefinedBayesianFusion
- R² = 0.5782 (提升 +0.0082)
- MAE = 4.31
- RMSE = 9.85

距离目标 (0.5800) 还差 0.0018

核心发现:
1. 贝叶斯启发的融合策略是当前最有效的单模型方法
2. 动态权重调整比固定权重更有效
3. 结合全局和局部信息比单纯使用一种信息更好

关键洞察:
- V6-Ensemble-V1等集成方法通过多个基础模型的加权组合达到R²=0.5803
- 单模型在捕捉CMAQ偏差和空间插值的动态关系方面有局限性
- 要达到R²=0.5800可能需要更强的非线性建模能力

代码文件:
- Round8_TrueInnovation.py: 初始创新方法
- Round9_GWRInnovation.py: GWR创新
- Round10_NeuralStyleFusion.py: 核感知融合
- Round11_OptimizedFusion.py: 贝叶斯融合优化
- Round12_FinalOptimization.py: 最终优化
"""

import pandas as pd

# 最终结果汇总
final_results = [
    {'method': 'RefinedBayesianFusion', 'R2': 0.5782, 'MAE': 4.31, 'RMSE': 9.85, 'MB': 0.05, 'improvement': 0.0082},
    {'method': 'BayesianInspiredFusion', 'R2': 0.5781, 'MAE': 4.31, 'RMSE': 9.85, 'MB': 0.04, 'improvement': 0.0081},
    {'method': 'CorrelationGuidedFusion', 'R2': 0.5773, 'MAE': 4.30, 'RMSE': 9.86, 'MB': 0.11, 'improvement': 0.0073},
    {'method': 'KernelAwareFusion', 'R2': 0.5770, 'MAE': 4.32, 'RMSE': 9.86, 'MB': 0.04, 'improvement': 0.0070},
    {'method': 'TripleBandwidthFusion', 'R2': 0.5765, 'MAE': 4.32, 'RMSE': 9.87, 'MB': 0.03, 'improvement': 0.0065},
    {'method': 'CMAQConditionedFusion', 'R2': 0.5759, 'MAE': 4.30, 'RMSE': 9.87, 'MB': 0.06, 'improvement': 0.0059},
    {'method': 'CMAQCentroidFusion', 'R2': 0.5757, 'MAE': 4.31, 'RMSE': 9.88, 'MB': 0.07, 'improvement': 0.0057},
    {'method': 'CMAQGuidedFusion', 'R2': 0.5756, 'MAE': 4.30, 'RMSE': 9.88, 'MB': 0.15, 'improvement': 0.0056},
    {'method': 'SpatialLocalWeighting', 'R2': 0.5744, 'MAE': 4.31, 'RMSE': 9.89, 'MB': 0.18, 'improvement': 0.0044},
]

df = pd.DataFrame(final_results)
df = df.sort_values('R2', ascending=False)
print("\n" + "="*70)
print("年平均数据融合 - 最终单模型创新结果")
print("="*70)
print(f"\nBaseline: VNA R2=0.5700")
print(f"Target: R2>=0.5800 (improvement>=0.01)")
print(f"\nBest single model: RefinedBayesianFusion R2=0.5782 (improvement+0.0082)")
print(f"Gap to target: {0.5800 - 0.5782:.4f}")
print("\n" + "-"*70)
print("All single model methods ranked by R2:")
print("-"*70)
print("-"*70)
for i, row in df.iterrows():
    print(f"  {row['method']:30s}: R2={row['R2']:.4f} (imp{row['improvement']:+.4f}), MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}")

print("\n" + "="*70)
print("代码文件位置: CodeWorkSpace/年均融合方法/")
print("="*70)