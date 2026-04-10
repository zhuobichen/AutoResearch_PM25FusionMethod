"""
年平均数据融合 - 最终单模型创新报告 (Round 13-18)
=================================================

项目目标:
- 基准: VNA R²=0.5700
- 目标: R² ≥ 0.5800 (提升 ≥ 0.01)
- 约束: 必须是单模型

=====================================
单模型创新方法排名 (Round 13-18)
=====================================

1. V2DualBiasFusion: R²=0.5794, MAE=4.31, RMSE=9.83
   提升: +0.0094
   原理: 双偏差校正融合，根据局部偏差类型动态调整aVNA/eVNA权重

2. FinalDualBiasFusion: R²=0.5792, MAE=4.31, RMSE=9.83
   提升: +0.0092
   原理: 双偏差校正融合的优化版本

3. EnhancedDualBiasFusion: R²=0.5792, MAE=4.31, RMSE=9.83
   提升: +0.0092
   原理: 增强双偏差校正融合

4. RefinedDualBiasFusion: R²=0.5791, MAE=4.31, RMSE=9.84
   提升: +0.0091

5. DualBiasFusion: R²=0.5790, MAE=4.31, RMSE=9.84
   提升: +0.0090
   原理: 双偏差校正融合

6. AdaptiveDualBiasFusion: R²=0.5789, MAE=4.32, RMSE=9.84
   提升: +0.0089
   原理: 结合局部相关性的自适应双偏差融合

7. SpatialStratifiedFusion: R²=0.5787, MAE=4.31, RMSE=9.84
   提升: +0.0087
   原理: 空间分层融合，根据地理位置偏差特征调整

8. OptimizedSpatialStratifiedFusion: R²=0.5784, MAE=4.31, RMSE=9.84
   提升: +0.0084

9. EnhancedSpatialStratifiedFusion: R²=0.5784, MAE=4.31, RMSE=9.84
   提升: +0.0084

10. ResidualNormalityCorrection: R²=0.5782
    提升: +0.0082

=====================================
历史最佳单模型 (Round 8-12)
=====================================

11. RefinedBayesianFusion: R²=0.5782
    原理: 贝叶斯启发的权重融合

12. BayesianInspiredFusion: R²=0.5781

13. CorrelationGuidedFusion: R²=0.5773

=====================================
结论
=====================================

最佳单模型: V2DualBiasFusion
- R² = 0.5794 (提升 +0.0094)
- MAE = 4.31
- RMSE = 9.83
- MB = 0.03

距离目标 (0.5800) 还差 0.0006

核心发现:
1. 双偏差校正融合是最有效的单模型方法
2. 根据局部偏差类型动态调整权重是关键
3. CMAQ偏差和局部空间特征结合效果最好

代码文件:
- Round13_SingleModelInnovation.py: 残差正态性校正、空间分层、分位数加权、自适应偏差校正
- Round14_AggressiveOptimization.py: 增强空间分层、局部相关性自适应、方差加权
- Round15_TrueInnovation.py: 残差克里金（计算太慢）、混合趋势克里金
- Round16_OptimizedFusion.py: 双偏差融合、自适应K近邻
- Round17_DualBiasOptimization.py: 双偏差融合深度优化
- Round18_FinalOptimization.py: 最终优化版本
"""

import pandas as pd

# 最终结果汇总
final_results = [
    {'method': 'V2DualBiasFusion', 'R2': 0.5794, 'MAE': 4.31, 'RMSE': 9.83, 'MB': 0.03, 'improvement': 0.0094},
    {'method': 'FinalDualBiasFusion', 'R2': 0.5792, 'MAE': 4.31, 'RMSE': 9.83, 'MB': 0.03, 'improvement': 0.0092},
    {'method': 'EnhancedDualBiasFusion', 'R2': 0.5792, 'MAE': 4.31, 'RMSE': 9.83, 'MB': 0.04, 'improvement': 0.0092},
    {'method': 'RefinedDualBiasFusion', 'R2': 0.5791, 'MAE': 4.31, 'RMSE': 9.84, 'MB': 0.04, 'improvement': 0.0091},
    {'method': 'DualBiasFusion', 'R2': 0.5790, 'MAE': 4.31, 'RMSE': 9.84, 'MB': 0.05, 'improvement': 0.0090},
    {'method': 'AdaptiveDualBiasFusion', 'R2': 0.5789, 'MAE': 4.32, 'RMSE': 9.84, 'MB': 0.02, 'improvement': 0.0089},
    {'method': 'SpatialStratifiedFusion', 'R2': 0.5787, 'MAE': 4.31, 'RMSE': 9.84, 'MB': 0.06, 'improvement': 0.0087},
    {'method': 'OptimizedSpatialStratifiedFusion', 'R2': 0.5784, 'MAE': 4.31, 'RMSE': 9.84, 'MB': 0.04, 'improvement': 0.0084},
    {'method': 'EnhancedSpatialStratifiedFusion', 'R2': 0.5784, 'MAE': 4.31, 'RMSE': 9.84, 'MB': 0.04, 'improvement': 0.0084},
    {'method': 'ResidualNormalityCorrection', 'R2': 0.5782, 'MAE': 4.31, 'RMSE': 9.85, 'MB': 0.05, 'improvement': 0.0082},
    {'method': 'AdaptiveBiasCorrection', 'R2': 0.5782, 'MAE': 4.30, 'RMSE': 9.85, 'MB': 0.11, 'improvement': 0.0082},
    {'method': 'VarianceWeightedFusion', 'R2': 0.5782, 'MAE': 4.29, 'RMSE': 9.85, 'MB': 0.13, 'improvement': 0.0082},
    {'method': 'RefinedBayesianFusion', 'R2': 0.5782, 'MAE': 4.31, 'RMSE': 9.85, 'MB': 0.05, 'improvement': 0.0082},
]

df = pd.DataFrame(final_results)
df = df.sort_values('R2', ascending=False)
print("\n" + "="*70)
print("年平均数据融合 - 最终单模型创新结果 (Round 13-18)")
print("="*70)
print(f"\nBaseline: VNA R2=0.5700")
print(f"Target: R2>=0.5800 (improvement>=0.01)")
print(f"\nBest single model: V2DualBiasFusion R2=0.5794 (improvement+0.0094)")
print(f"Gap to target: {0.5800 - 0.5794:.4f}")
print("\n" + "-"*70)
print("All single model methods ranked by R2:")
print("-"*70)
for i, row in df.iterrows():
    print(f"  {row['method']:35s}: R2={row['R2']:.4f} (imp{row['improvement']:+.4f}), MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}")

print("\n" + "="*70)
print("代码文件位置: CodeWorkSpace/年均融合方法/")
print("="*70)