"""
1月整月多日验证脚本
对比 CSP-RK 和 PolyRK 在1月每一天的表现
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import pandas as pd
import numpy as np
from datetime import datetime
import netCDF4 as nc

# 导入方法
from CodeWorkSpace.新融合方法代码.PolyRK import run_poly_rk_ten_fold
from CodeWorkSpace.新融合方法代码.CSPRK import run_csprk_ten_fold

def main():
    print("=" * 60)
    print("1月整月验证: CSP-RK vs PolyRK")
    print("=" * 60)

    # 获取1月所有日期
    monitor_df = pd.read_csv('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
    dates = monitor_df['Date'].unique()
    jan_dates = sorted([d for d in dates if d.startswith('2020-01')])

    print(f"\n1月共 {len(jan_dates)} 天\n")

    results = []

    for i, day in enumerate(jan_dates):
        print(f"[{i+1}/{len(jan_dates)}] {day}...", end=" ", flush=True)

        try:
            # PolyRK
            _, rkpoly_result = run_poly_rk_ten_fold(day)
            print(f"RK={rkpoly_result['R2']:.4f}", end=" ", flush=True)

            # CSP-RK
            _, csprk_result = run_csprk_ten_fold(day)
            print(f"CSP={csprk_result['R2']:.4f}", end=" ", flush=True)

            improvement = csprk_result['R2'] - rkpoly_result['R2']
            print(f"Δ={improvement:+.4f}")

            results.append({
                'Date': day,
                'RKPoly_R2': rkpoly_result['R2'],
                'RKPoly_MAE': rkpoly_result['MAE'],
                'RKPoly_RMSE': rkpoly_result['RMSE'],
                'CSPRK_R2': csprk_result['R2'],
                'CSPRK_MAE': csprk_result['MAE'],
                'CSPRK_RMSE': csprk_result['RMSE'],
                'R2_Improvement': improvement
            })
        except Exception as e:
            print(f"错误: {e}")
            continue

    # 汇总
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)

    rkpoly_r2 = [r['RKPoly_R2'] for r in results]
    csprk_r2 = [r['CSPRK_R2'] for r in results]
    improvements = [r['R2_Improvement'] for r in results]

    print(f"\nPolyRK:")
    print(f"  平均 R²: {np.mean(rkpoly_r2):.4f} ± {np.std(rkpoly_r2):.4f}")
    print(f"  最小 R²: {np.min(rkpoly_r2):.4f}")
    print(f"  最大 R²: {np.max(rkpoly_r2):.4f}")

    print(f"\nCSP-RK:")
    print(f"  平均 R²: {np.mean(csprk_r2):.4f} ± {np.std(csprk_r2):.4f}")
    print(f"  最小 R²: {np.min(csprk_r2):.4f}")
    print(f"  最大 R²: {np.max(csprk_r2):.4f}")

    print(f"\nCSP-RK 相对 PolyRK 的 R² 提升:")
    print(f"  平均提升: {np.mean(improvements):+.4f}")
    print(f"  提升天数: {sum(1 for x in improvements if x > 0)}/{len(improvements)}")
    print(f"  提升比例: {sum(1 for x in improvements if x > 0)/len(improvements)*100:.1f}%")

    # 判定
    avg_improvement = np.mean(improvements)
    print(f"\n判定结果:")
    if avg_improvement >= 0.01:
        print(f"  ✅ 通过！平均 R² 提升 {avg_improvement:.4f} >= 0.01")
    elif avg_improvement >= 0:
        print(f"  ⚠️ 有提升但不显著。平均 R² 提升 {avg_improvement:.4f} < 0.01")
    else:
        print(f"  ❌ 未通过。CSP-RK 反而更差。平均 R² 变化 {avg_improvement:.4f}")

    # 保存详细结果
    output_file = 'test_result/创新方法/january_month_validation.csv'
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\n详细结果已保存到: {output_file}")

if __name__ == '__main__':
    main()
