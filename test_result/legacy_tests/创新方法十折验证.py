"""
创新方法十折交叉验证
====================
对MSAK和STRK进行十折交叉验证，计算R2, MAE, RMSE, MB指标
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import netCDF4 as nc
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

root_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
cmaq_file = f'{root_dir}/test_data/raw/CMAQ/2020_PM25.nc'
monitor_file = f'{root_dir}/test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv'
fold_file = f'{root_dir}/test_data/fold_split_table.csv'
output_dir = f'{root_dir}/test_result/创新方法'
os.makedirs(output_dir, exist_ok=True)

# 导入MSAK和STRK
sys.path.insert(0, f'{root_dir}/CodeWorkSpace/新融合方法代码')
from MSAK import MSAK, cross_validate as msak_cross_validate, get_cmaq_at_site
from STRK import STRK, cross_validate as strk_cross_validate


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MB': np.mean(y_pred - y_true)
    }


def run_ten_fold_validation(method_class, method_name, selected_day='2020-01-01'):
    """
    执行十折交叉验证

    参数:
        method_class: 方法类 (MSAK或STRK)
        method_name: 方法名称字符串
        selected_day: 测试日期

    返回:
        dict: 包含R2, MAE, RMSE, MB
    """
    print("=" * 60)
    print(f"{method_name} Ten-Fold Cross Validation")
    print("=" * 60)

    # 加载CMAQ数据
    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    # 筛选日期
    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    if 0 <= day_idx < pred_pm25.shape[0]:
        pm25_day = pred_pm25[day_idx]
    else:
        print(f"Error: day_idx {day_idx} out of range")
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}

    # 提取站点CMAQ值
    cmaq_vals = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pm25_day)
        cmaq_vals.append(val)
    day_df['CMAQ'] = cmaq_vals

    print(f"Date: {selected_day}, Total sites: {len(day_df)}")

    all_y_true = []
    all_y_pred = []

    # 十折验证
    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0:
            print(f"  Fold {fold_id}: No test data, skip")
            continue

        # 创建模型实例
        if method_name == 'MSAK':
            model = MSAK(lambda0=15.0, beta=0.8, sigma_pg=1.2,
                         pg_crit=3.0, pg_mid=2.5, gamma=1.5, n_scales=3)
            ws_col = 'WS' if 'WS' in train_df.columns else None
            model.fit(train_df, ws_col=ws_col)
            result = model.predict(test_df, ws_col=ws_col)
        else:  # STRK
            model = STRK(lambda_s=20.0, tau=3.0, rho_s=0.5,
                         theta1=0.3, theta2=0.15, theta3=0.25)
            model.fit(train_df)
            result = model.predict(test_df)

        all_y_true.extend(test_df['Conc'].values)
        all_y_pred.extend(result['fusion'])

        fold_metrics = compute_metrics(test_df['Conc'].values, result['fusion'])
        print(f"  Fold {fold_id}: R2={fold_metrics['R2']:.4f}, MAE={fold_metrics['MAE']:.2f}, RMSE={fold_metrics['RMSE']:.2f}")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    metrics = compute_metrics(all_y_true, all_y_pred)
    print(f"\n=== {method_name} Final Results ===")
    print(f"  R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

    return metrics


def main():
    print("\n" + "=" * 70)
    print("INNOVATIVE METHODS TEN-FOLD CROSS VALIDATION")
    print("=" * 70)

    selected_day = '2020-01-01'

    # ========== MSAK验证 ==========
    print("\n" + "#" * 70)
    print("# MSAK (Multi-Scale Stability-Adaptive Kriging)")
    print("#" * 70)
    msak_metrics = run_ten_fold_validation(MSAK, 'MSAK', selected_day)

    # 保存MSAK结果
    msak_summary = pd.DataFrame([{
        'Method': 'MSAK',
        'R2': msak_metrics['R2'],
        'MAE': msak_metrics['MAE'],
        'RMSE': msak_metrics['RMSE'],
        'MB': msak_metrics['MB']
    }])
    msak_summary.to_csv(f'{output_dir}/MSAK_summary.csv', index=False)
    print(f"\nMSAK results saved to: {output_dir}/MSAK_summary.csv")

    # ========== STRK验证 ==========
    print("\n" + "#" * 70)
    print("# STRK (Spatio-Temporal Residual Co-Kriging)")
    print("#" * 70)
    strk_metrics = run_ten_fold_validation(STRK, 'STRK', selected_day)

    # 保存STRK结果
    strk_summary = pd.DataFrame([{
        'Method': 'STRK',
        'R2': strk_metrics['R2'],
        'MAE': strk_metrics['MAE'],
        'RMSE': strk_metrics['RMSE'],
        'MB': strk_metrics['MB']
    }])
    strk_summary.to_csv(f'{output_dir}/STRK_summary.csv', index=False)
    print(f"\nSTRK results saved to: {output_dir}/STRK_summary.csv")

    # ========== 生成对比报告 ==========
    print("\n" + "#" * 70)
    print("# COMPARISON REPORT")
    print("#" * 70)

    # Benchmark参考值
    rk_poly_r2 = 0.8519
    rk_poly_mae = 7.09
    rk_poly_rmse = 11.05
    target_r2 = 0.8619

    report_content = f"""# 创新方法十折交叉验证对比报告

## 评测日期
2026-04-09

## 评测方法
- **测试日期**: {selected_day}
- **验证策略**: 十折交叉验证 (10-Fold Cross Validation)

## 基准方法
| 方法 | R2 | MAE | RMSE | MB |
|------|-----|-----|------|-----|
| RK-Poly (当前最佳) | {rk_poly_r2:.4f} | {rk_poly_mae:.2f} | {rk_poly_rmse:.2f} | - |

**目标**: R2 >= {target_r2:.4f} (相对于RK-Poly提升0.01)

## 评测方法结果

### MSAK (多尺度稳定度自适应克里金)
| 指标 | 值 |
|------|-----|
| R2 | {msak_metrics['R2']:.4f} |
| MAE | {msak_metrics['MAE']:.2f} |
| RMSE | {msak_metrics['RMSE']:.2f} |
| MB | {msak_metrics['MB']:.2f} |

### STRK (时空残差共克里金)
| 指标 | 值 |
|------|-----|
| R2 | {strk_metrics['R2']:.4f} |
| MAE | {strk_metrics['MAE']:.2f} |
| RMSE | {strk_metrics['RMSE']:.2f} |
| MB | {strk_metrics['MB']:.2f} |

## 方法对比

| 方法 | R2 | MAE | RMSE | MB | 相对于RK-Poly R2变化 |
|------|-----|-----|------|-----|---------------------|
| RK-Poly | {rk_poly_r2:.4f} | {rk_poly_mae:.2f} | {rk_poly_rmse:.2f} | - | - |
| MSAK | {msak_metrics['R2']:.4f} | {msak_metrics['MAE']:.2f} | {msak_metrics['RMSE']:.2f} | {msak_metrics['MB']:.2f} | {msak_metrics['R2'] - rk_poly_r2:+.4f} |
| STRK | {strk_metrics['R2']:.4f} | {strk_metrics['MAE']:.2f} | {strk_metrics['RMSE']:.2f} | {strk_metrics['MB']:.2f} | {strk_metrics['R2'] - rk_poly_r2:+.4f} |

## 结论

### MSAK
- R2 = {msak_metrics['R2']:.4f} (目标: >= {target_r2:.4f})
- 状态: {'达标' if msak_metrics['R2'] >= target_r2 else '未达标'}

### STRK
- R2 = {strk_metrics['R2']:.4f} (目标: >= {target_r2:.4f})
- 状态: {'达标' if strk_metrics['R2'] >= target_r2 else '未达标'}

"""

    # 写入对比报告
    report_path = f'{output_dir}/comparison_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(report_content)
    print(f"\n对比报告已保存到: {report_path}")

    return msak_metrics, strk_metrics


if __name__ == '__main__':
    main()
