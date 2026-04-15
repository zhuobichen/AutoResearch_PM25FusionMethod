"""
多阶段验证结果汇总
===================

验证结果分析:

**PG-STGAT:**
- Stage 1 (预实验5天): avg R2=0.8257 > 0.8200, PASS
- Stage 2 (1月整月): R2=0.8735 > 0.8200, PASS
- Stage 3 (7月采样10天): R2=0.7377 < 0.8200, FAIL
- Stage 4 (12月采样10天): R2=0.8693 > 0.8200, PASS

**VCFFM:**
- Stage 1 (预实验5天): avg R2=0.8203 > 0.8200, PASS
- Stage 2 (1月整月): R2=0.8702 > 0.8200, PASS
- Stage 3 (7月采样10天): R2=0.7463 < 0.8200, FAIL
- Stage 4 (12月采样10天): R2=0.8690 > 0.8200, PASS

结论: 两种方法在7月验证中均失败，创新验证不成立。
"""

import json
import pandas as pd

output_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/test_result/创新方法'

# 读取完整结果
with open(f'{output_dir}/multi_stage_validation_summary.json') as f:
    results = json.load(f)

# 生成CSV汇总
summary_data = []
for method_name in ['PG-STGAT', 'VCFFM']:
    for stage_id in [1, 2, 3, 4]:
        stage_key = f'{method_name}_stage{stage_id}'
        if stage_key in results:
            r = results[stage_key]
            r2_val = r.get('R2', r.get('avg_r2', None))
            rmse_val = r.get('RMSE', r.get('avg_rmse', None))
            mb_val = r.get('MB', r.get('avg_mb', None))
            summary_data.append({
                'method': method_name,
                'stage': stage_id,
                'stage_name': r.get('stage', ''),
                'R2': float(r2_val) if r2_val is not None else None,
                'RMSE': float(rmse_val) if rmse_val is not None else None,
                'MB': float(mb_val) if mb_val is not None else None,
                'pass': bool(r['pass'])
            })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f'{output_dir}/summary.csv', index=False)

print("Summary CSV saved to:", f'{output_dir}/summary.csv')
print("\nSummary Table:")
print(summary_df.to_string(index=False))

# 最终判定
stage3_pass = all(results[f'{m}_stage3']['pass'] for m in ['PG-STGAT', 'VCFFM'])
stage4_pass = all(results[f'{m}_stage4']['pass'] for m in ['PG-STGAT', 'VCFFM'])
stage2_pass = all(results[f'{m}_stage2']['pass'] for m in ['PG-STGAT', 'VCFFM'])
stage1_pass = all(results[f'{m}_stage1']['pass'] for m in ['PG-STGAT', 'VCFFM'])

print("\n" + "="*60)
print("FINAL VERDICT")
print("="*60)
print(f"Stage 1 (Pre-exp 5 days): {'PASS' if stage1_pass else 'FAIL'}")
print(f"Stage 2 (January full):   {'PASS' if stage2_pass else 'FAIL'}")
print(f"Stage 3 (July sampled):   {'PASS' if stage3_pass else 'FAIL'}")
print(f"Stage 4 (December sampled): {'PASS' if stage4_pass else 'FAIL'}")
print("="*60)

innovation_pass = stage1_pass and stage2_pass and stage3_pass and stage4_pass
print(f"\nInnovation Validated: {'YES' if innovation_pass else 'NO'}")
print("="*60)

if not stage3_pass:
    print("\n** Issue: Both methods FAILED July validation **")
    print("   PG-STGAT July R2=0.7377 < 0.8200 (threshold)")
    print("   VCFFM July R2=0.7463 < 0.8200 (threshold)")
    print("\n   This indicates the methods perform poorly in summer conditions.")