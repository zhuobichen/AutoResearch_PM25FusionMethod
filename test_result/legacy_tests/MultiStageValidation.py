# -*- coding: utf-8 -*-
"""
Multi-Stage Innovation Validation for PM2.5 CMAQ Fusion Methods
================================================================
Implements the strict multi-stage validation:
- Stage 1: Pre-experiment (5 days): 2020-01-01 ~ 2020-01-05
- Stage 2: January full month (31 days)
- Stage 3: July full month (31 days)
- Stage 4: December full month (31 days)

Innovation conditions (MUST ALL BE SATISFIED):
- R2 >= 0.8200
- RMSE <= 12.52
- |MB| <= 0.08

Uses the original RRK implementation from RRK.py

Author: Claude Agent
Date: 2026-04-10
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import netCDF4 as nc

root_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
cmaq_file = f'{root_dir}/test_data/raw/CMAQ/2020_PM25.nc'
monitor_file = f'{root_dir}/test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv'
fold_file = f'{root_dir}/test_data/fold_split_table.csv'
output_dir = f'{root_dir}/test_result/cross_day_validation'
os.makedirs(output_dir, exist_ok=True)

# Innovation thresholds
R2_THRESHOLD = 0.8200
RMSE_THRESHOLD = 12.52
MB_THRESHOLD = 0.08


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics"""
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


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    """Get CMAQ value at site location (nearest neighbor)"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def run_rrk_ten_fold(day_df, poly_degree=2, huber_delta=1.35):
    """
    Run RRK (Robust Rescaled Kriging) ten-fold cross validation
    Uses the correct RRK implementation with polynomial features and Huber regression
    """
    kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
              RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

    all_y_true = []
    all_y_pred = []

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0:
            continue

        X_train = train_df[['Lon', 'Lat']].values
        X_test = test_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values
        y_test = test_df['Conc'].values
        m_train = train_df['CMAQ'].values
        m_test = test_df['CMAQ'].values

        # Polynomial features
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        # Huber robust polynomial calibration
        huber = HuberRegressor(epsilon=huber_delta, max_iter=1000)
        huber.fit(m_train_poly, y_train)
        pred_huber = huber.predict(m_test_poly)
        residual_huber = y_train - huber.predict(m_train_poly)

        # GPR on Huber residuals
        gpr_huber = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_huber.fit(X_train, residual_huber)
        gpr_huber_pred, _ = gpr_huber.predict(X_test, return_std=True)

        # Final prediction: Huber poly + GPR residual
        final_pred = pred_huber + gpr_huber_pred

        all_y_true.extend(y_test)
        all_y_pred.extend(final_pred)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    return compute_metrics(all_y_true, all_y_pred)


def load_data_for_day(selected_day):
    """Load all required data for a specific day"""
    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    if 0 <= day_idx < pred_pm25.shape[0]:
        pm25_day = pred_pm25[day_idx]
    else:
        return None, None, None, None

    # Extract CMAQ at sites
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pm25_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    return day_df, lon_cmaq, lat_cmaq, pm25_day


def run_stage_validation(method_name, day_list, stage_name):
    """
    Run validation for a stage (pre-experiment, 1月, 7月, 12月)
    """
    print(f"\n{'='*70}")
    print(f"Stage: {stage_name}")
    print(f"Method: {method_name}")
    print(f"Days: {len(day_list)}")
    print(f"{'='*70}")

    all_r2 = []
    all_rmse = []
    all_mae = []
    all_mb = []
    daily_results = []

    for day in day_list:
        try:
            day_df, _, _, _ = load_data_for_day(day)
            if day_df is None or len(day_df) < 10:
                print(f"  {day}: Insufficient data, skip")
                continue

            metrics = run_rrk_ten_fold(day_df)
            all_r2.append(metrics['R2'])
            all_rmse.append(metrics['RMSE'])
            all_mae.append(metrics['MAE'])
            all_mb.append(metrics['MB'])
            daily_results.append({
                'date': day,
                'R2': metrics['R2'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MB': metrics['MB']
            })
            print(f"  {day}: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")
        except Exception as e:
            print(f"  {day}: Error - {str(e)[:60]}")
            continue

    if len(all_r2) == 0:
        return {
            'stage': stage_name,
            'n_days': 0,
            'mean_R2': np.nan,
            'mean_RMSE': np.nan,
            'mean_MAE': np.nan,
            'mean_MB': np.nan,
            'mean_abs_MB': np.nan,
            'r2_pass': False,
            'rmse_pass': False,
            'mb_pass': False,
            'pass': False,
            'daily_results': []
        }

    mean_r2 = np.mean(all_r2)
    mean_rmse = np.mean(all_rmse)
    mean_mae = np.mean(all_mae)
    mean_mb = np.mean(all_mb)
    mean_abs_mb = np.abs(mean_mb)

    # Check all 3 conditions
    r2_pass = mean_r2 >= R2_THRESHOLD
    rmse_pass = mean_rmse <= RMSE_THRESHOLD
    mb_pass = mean_abs_mb <= MB_THRESHOLD
    pass_all = r2_pass and rmse_pass and mb_pass

    print(f"\n  === Stage Summary ===")
    print(f"  Days tested: {len(all_r2)}")
    print(f"  Mean R2: {mean_r2:.4f} {'[PASS]' if r2_pass else '[FAIL]'} (threshold: {R2_THRESHOLD})")
    print(f"  Mean RMSE: {mean_rmse:.2f} {'[PASS]' if rmse_pass else '[FAIL]'} (threshold: {RMSE_THRESHOLD})")
    print(f"  Mean MB: {mean_mb:.4f}, |MB|: {mean_abs_mb:.4f} {'[PASS]' if mb_pass else '[FAIL]'} (threshold: {MB_THRESHOLD})")
    print(f"  Overall: {'PASS' if pass_all else 'FAIL'}")

    return {
        'stage': stage_name,
        'n_days': len(all_r2),
        'mean_R2': mean_r2,
        'mean_RMSE': mean_rmse,
        'mean_MAE': mean_mae,
        'mean_MB': mean_mb,
        'mean_abs_MB': mean_abs_mb,
        'r2_pass': r2_pass,
        'rmse_pass': rmse_pass,
        'mb_pass': mb_pass,
        'pass': pass_all,
        'daily_results': daily_results
    }


def generate_report(method_name, stage_results, output_path):
    """Generate markdown report for the validation"""
    report = f"""# {method_name} Multi-Stage Innovation Validation Report

Generated: 2026-04-10

## Innovation Verification Criteria

| Metric | Requirement | Benchmark |
|--------|-------------|-----------|
| R2 | >= 0.8200 | eVNA=0.8100 + 0.01 |
| RMSE | <= 12.52 | eVNA baseline |
| |MB| | <= 0.08 | eVNA baseline |

## Stage Results

### Stage 1: Pre-experiment (5 days: 2020-01-01 ~ 2020-01-05)
"""

    s1 = stage_results['stage1']
    report += f"""
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Mean R2 | {s1['mean_R2']:.4f} | >= 0.8200 | {'PASS' if s1['r2_pass'] else 'FAIL'} |
| Mean RMSE | {s1['mean_RMSE']:.2f} | <= 12.52 | {'PASS' if s1['rmse_pass'] else 'FAIL'} |
| Mean |MB| | {s1['mean_abs_MB']:.4f} | <= 0.08 | {'PASS' if s1['mb_pass'] else 'FAIL'} |
| Days Tested | {s1['n_days']} | - | - |
| **Overall** | - | - | **{'PASS' if s1['pass'] else 'FAIL'}** |

### Daily Breakdown

| Date | R2 | RMSE | MB |
|------|-----|------|-----|
"""
    for d in s1.get('daily_results', []):
        report += f"| {d['date']} | {d['R2']:.4f} | {d['RMSE']:.2f} | {d['MB']:.4f} |\n"

    report += "\n### Stage 2: January Full Month (31 days)\n\n"
    s2 = stage_results['stage2']
    if s2['n_days'] > 0:
        report += f"""
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Mean R2 | {s2['mean_R2']:.4f} | >= 0.8200 | {'PASS' if s2['r2_pass'] else 'FAIL'} |
| Mean RMSE | {s2['mean_RMSE']:.2f} | <= 12.52 | {'PASS' if s2['rmse_pass'] else 'FAIL'} |
| Mean |MB| | {s2['mean_abs_MB']:.4f} | <= 0.08 | {'PASS' if s2['mb_pass'] else 'FAIL'} |
| Days Tested | {s2['n_days']} | - | - |
| **Overall** | - | - | **{'PASS' if s2['pass'] else 'FAIL'}** |

"""
    else:
        report += "Not tested (Stage 1 not passed)\n\n"

    report += "### Stage 3: July Full Month (31 days)\n\n"
    s3 = stage_results['stage3']
    if s3['n_days'] > 0:
        report += f"""
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Mean R2 | {s3['mean_R2']:.4f} | >= 0.8200 | {'PASS' if s3['r2_pass'] else 'FAIL'} |
| Mean RMSE | {s3['mean_RMSE']:.2f} | <= 12.52 | {'PASS' if s3['rmse_pass'] else 'FAIL'} |
| Mean |MB| | {s3['mean_abs_MB']:.4f} | <= 0.08 | {'PASS' if s3['mb_pass'] else 'FAIL'} |
| Days Tested | {s3['n_days']} | - | - |
| **Overall** | - | - | **{'PASS' if s3['pass'] else 'FAIL'}** |

"""
    else:
        report += "Not tested (Stage 2 not passed)\n\n"

    report += "### Stage 4: December Full Month (31 days)\n\n"
    s4 = stage_results['stage4']
    if s4['n_days'] > 0:
        report += f"""
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Mean R2 | {s4['mean_R2']:.4f} | >= 0.8200 | {'PASS' if s4['r2_pass'] else 'FAIL'} |
| Mean RMSE | {s4['mean_RMSE']:.2f} | <= 12.52 | {'PASS' if s4['rmse_pass'] else 'FAIL'} |
| Mean |MB| | {s4['mean_abs_MB']:.4f} | <= 0.08 | {'PASS' if s4['mb_pass'] else 'FAIL'} |
| Days Tested | {s4['n_days']} | - | - |
| **Overall** | - | - | **{'PASS' if s4['pass'] else 'FAIL'}** |

"""
    else:
        report += "Not tested (Stage 3 not passed)\n\n"

    # Overall conclusion
    overall_pass = s1['pass'] and s2['pass'] and s3['pass'] and s4['pass']
    report += "## Conclusion\n\n"
    report += f"**Innovation Verification: {'PASSED - ALL 4 STAGES COMPLETED' if overall_pass else 'FAILED'}**\n\n"

    if overall_pass:
        report += f"""### {method_name} is verified as an innovative method

- Pre-experiment (5 days): R2={s1['mean_R2']:.4f} >= 0.8200 [PASS]
- January (31 days): R2={s2['mean_R2']:.4f} >= 0.8200 [PASS]
- July (31 days): R2={s3['mean_R2']:.4f} >= 0.8200 [PASS]
- December (31 days): R2={s4['mean_R2']:.4f} >= 0.8200 [PASS]

All 4 stages passed. Innovation is verified.
"""
    else:
        failed_stages = []
        if not s1['pass']:
            failed_stages.append("Stage 1 (Pre-experiment)")
        if not s2['pass']:
            failed_stages.append("Stage 2 (January)")
        if not s3['pass']:
            failed_stages.append("Stage 3 (July)")
        if not s4['pass']:
            failed_stages.append("Stage 4 (December)")
        report += f"""### Innovation verification failed

Failed stages: {', '.join(failed_stages)}

Innovation is NOT verified.
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")
    return report


def main():
    """Main function"""
    print("="*70)
    print("Multi-Stage Innovation Validation for RRK")
    print("="*70)
    print(f"\nThresholds:")
    print(f"  R2 >= {R2_THRESHOLD}")
    print(f"  RMSE <= {RMSE_THRESHOLD}")
    print(f"  |MB| <= {MB_THRESHOLD}")

    method_name = "RRK"

    # Define day ranges for each stage
    # Stage 1: Pre-experiment (5 days)
    stage1_days = [f'2020-01-{i:02d}' for i in range(1, 6)]

    # Stage 2: January full month (31 days)
    stage2_days = [f'2020-01-{i:02d}' for i in range(1, 32)]

    # Stage 3: July full month (31 days)
    stage3_days = [f'2020-07-{i:02d}' for i in range(1, 32)]

    # Stage 4: December full month (31 days)
    stage4_days = [f'2020-12-{i:02d}' for i in range(1, 32)]

    stage_results = {}

    # Run Stage 1: Pre-experiment
    stage1_result = run_stage_validation(method_name, stage1_days, "Pre-experiment (5 days: 2020-01-01 ~ 2020-01-05)")
    stage_results['stage1'] = stage1_result

    # Stage 2: Only if Stage 1 passed
    if stage1_result['pass']:
        print("\n\nStage 1 PASSED. Proceeding to Stage 2...")
        stage2_result = run_stage_validation(method_name, stage2_days, "January Full Month (31 days)")
        stage_results['stage2'] = stage2_result
    else:
        print("\n\nStage 1 FAILED. Stopping pipeline.")
        stage_results['stage2'] = {'n_days': 0, 'pass': False, 'mean_R2': np.nan, 'mean_RMSE': np.nan, 'mean_MB': np.nan, 'mean_abs_MB': np.nan, 'r2_pass': False, 'rmse_pass': False, 'mb_pass': False}

    # Stage 3: Only if Stage 2 passed
    if stage_results['stage2'].get('pass', False):
        print("\n\nStage 2 PASSED. Proceeding to Stage 3...")
        stage3_result = run_stage_validation(method_name, stage3_days, "July Full Month (31 days)")
        stage_results['stage3'] = stage3_result
    else:
        print("\n\nStage 2 FAILED. Stopping pipeline.")
        stage_results['stage3'] = {'n_days': 0, 'pass': False, 'mean_R2': np.nan, 'mean_RMSE': np.nan, 'mean_MB': np.nan, 'mean_abs_MB': np.nan, 'r2_pass': False, 'rmse_pass': False, 'mb_pass': False}

    # Stage 4: Only if Stage 3 passed
    if stage_results['stage3'].get('pass', False):
        print("\n\nStage 3 PASSED. Proceeding to Stage 4...")
        stage4_result = run_stage_validation(method_name, stage4_days, "December Full Month (31 days)")
        stage_results['stage4'] = stage4_result
    else:
        print("\n\nStage 3 FAILED. Stopping pipeline.")
        stage_results['stage4'] = {'n_days': 0, 'pass': False, 'mean_R2': np.nan, 'mean_RMSE': np.nan, 'mean_MB': np.nan, 'mean_abs_MB': np.nan, 'r2_pass': False, 'rmse_pass': False, 'mb_pass': False}

    # Generate report
    report_path = os.path.join(output_dir, f'{method_name}_multi_stage_report.md')
    report = generate_report(method_name, stage_results, report_path)

    # Save summary CSV
    s1 = stage_results['stage1']
    s2 = stage_results['stage2']
    s3 = stage_results['stage3']
    s4 = stage_results['stage4']

    summary_data = {
        'method': method_name,
        'stage1_n_days': s1.get('n_days', 0),
        'stage1_r2': s1.get('mean_R2', np.nan),
        'stage1_rmse': s1.get('mean_RMSE', np.nan),
        'stage1_mb': s1.get('mean_MB', np.nan),
        'stage1_abs_mb': s1.get('mean_abs_MB', np.nan),
        'stage1_r2_pass': s1.get('r2_pass', False),
        'stage1_rmse_pass': s1.get('rmse_pass', False),
        'stage1_mb_pass': s1.get('mb_pass', False),
        'stage1_pass': s1.get('pass', False),
        'stage2_n_days': s2.get('n_days', 0),
        'stage2_r2': s2.get('mean_R2', np.nan),
        'stage2_rmse': s2.get('mean_RMSE', np.nan),
        'stage2_mb': s2.get('mean_MB', np.nan),
        'stage2_abs_mb': s2.get('mean_abs_MB', np.nan),
        'stage2_pass': s2.get('pass', False),
        'stage3_n_days': s3.get('n_days', 0),
        'stage3_r2': s3.get('mean_R2', np.nan),
        'stage3_rmse': s3.get('mean_RMSE', np.nan),
        'stage3_mb': s3.get('mean_MB', np.nan),
        'stage3_abs_mb': s3.get('mean_abs_MB', np.nan),
        'stage3_pass': s3.get('pass', False),
        'stage4_n_days': s4.get('n_days', 0),
        'stage4_r2': s4.get('mean_R2', np.nan),
        'stage4_rmse': s4.get('mean_RMSE', np.nan),
        'stage4_mb': s4.get('mean_MB', np.nan),
        'stage4_abs_mb': s4.get('mean_abs_MB', np.nan),
        'stage4_pass': s4.get('pass', False),
        'all_stages_pass': s1.get('pass', False) and s2.get('pass', False) and s3.get('pass', False) and s4.get('pass', False)
    }

    summary_df = pd.DataFrame([summary_data])
    csv_path = os.path.join(output_dir, f'{method_name}_multi_stage_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved to: {csv_path}")

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

    return stage_results


if __name__ == '__main__':
    results = main()