#!/usr/bin/env python
"""
Benchmark Test for v23 - Conservative optimization based on v20

V23 uses the same default parameters as V20:
- numit: 2500
- burn: 500
- thin: 1
- neighbor: 3
- cmaqres: 12

Expected Results (matching V20):
- R2 >= 0.998
- MB ~= 0
- RMSE target <= 0.4 (vs C# baseline)
"""
import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

# Add v23 to path
sys.path.insert(0, 'E:/CodeProject/DataFusion/python_downscaler_v23')
from common_setting import CommonSetting
from pm.calculation.pm25_downscaler_calculator import PM25DownscalerCalculator


def load_test_data():
    """Load test data from benchmark directory."""
    base_dir = Path('E:/CodeProject/DataFusion/benchmark_test')
    monitor_df = pd.read_csv(base_dir / 'monitor.csv')
    model_df = pd.read_csv(base_dir / 'model.csv')

    # Extract data
    matrix_latlon_model = model_df[['Latitude', 'Longitude']].values
    matrix_model = model_df[['PM25']].values

    matrix_latlon_monitor = monitor_df[['Latitude', 'Longitude']].values
    matrix_monitor = monitor_df[['PM25']].values

    return matrix_latlon_model, matrix_latlon_monitor, matrix_model, matrix_monitor


def compute_metrics(ybar, sepred, baseline_ybar, baseline_sepred):
    """Compute comparison metrics."""
    metrics = {}

    # R2 for ybar
    ss_res = np.sum((ybar - baseline_ybar) ** 2)
    ss_tot = np.sum((baseline_ybar - np.mean(baseline_ybar)) ** 2)
    metrics['R2_ybar'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # R2 for sepred
    ss_res_s = np.sum((sepred - baseline_sepred) ** 2)
    ss_tot_s = np.sum((baseline_sepred - np.mean(baseline_sepred)) ** 2)
    metrics['R2_sepred'] = 1 - (ss_res_s / ss_tot_s) if ss_tot_s != 0 else 0.0

    # RMSE
    diff = ybar - baseline_ybar
    metrics['RMSE'] = float(np.sqrt(np.mean(diff ** 2)))

    # MB (Mean Bias)
    metrics['MB'] = float(np.mean(diff))

    # MaxAE (Maximum Absolute Error)
    max_abs_error = float(np.max(np.abs(diff)))
    baseline_range = np.max(baseline_ybar) - np.min(baseline_ybar)
    if baseline_range != 0:
        metrics['MaxAE_pct'] = (max_abs_error / baseline_range) * 100
    else:
        metrics['MaxAE_pct'] = 0.0

    metrics['MaxAE'] = max_abs_error

    return metrics


def main():
    print("=" * 70)
    print("BENCHMARK TEST: v23 (Conservative optimization based on v20)")
    print("=" * 70)
    print()
    print("Parameters: numit=2500, burn=500, thin=1, neighbor=3, cmaqres=12")
    print()

    # Load data
    print("Loading test data...")
    matrix_latlon_model, matrix_latlon_monitor, matrix_model, matrix_monitor = load_test_data()
    print(f"Data: {len(matrix_latlon_model)} grids, {len(matrix_latlon_monitor)} monitors")

    # Load C# baseline (v18 is the stable baseline)
    baseline_csv = Path('E:/CodeProject/DataFusion/benchmark_test/v18_csharp_output.csv')
    cs_df = pd.read_csv(baseline_csv)
    cs_ybar = cs_df['PM25_Prediction'].values
    cs_sepred = cs_df['PM25_StandardError'].values
    print(f"C# baseline loaded: {len(cs_ybar)} points")
    print(f"C# ybar range: [{cs_ybar.min():.4f}, {cs_ybar.max():.4f}]")

    # Run v23
    print("\nRunning v23...")
    setting = CommonSetting()
    start_time = time.time()

    try:
        py_ybar, py_sepred = PM25DownscalerCalculator.run(
            matrix_latlon_model=matrix_latlon_model.astype(float),
            matrix_latlon_monitor=matrix_latlon_monitor.astype(float),
            matrix_model=matrix_model.astype(float),
            matrix_monitor=matrix_monitor.astype(float),
            setting=setting,
            seed=42,
        )
        elapsed_ms = (time.time() - start_time) * 1000
    except Exception as e:
        print(f"ERROR: v23 failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Execution time: {elapsed_ms:.2f}ms")
    print(f"Python v23 output: ybar size={len(py_ybar)}, sepred size={len(py_sepred)}")
    print(f"Python v23 ybar range: [{py_ybar.min():.4f}, {py_ybar.max():.4f}]")

    # Check length match
    if len(py_ybar) != len(cs_ybar):
        print(f"\nWARNING: Output length mismatch! Python={len(py_ybar)}, C#={len(cs_ybar)}")
        min_len = min(len(py_ybar), len(cs_ybar))
        py_ybar = py_ybar[:min_len]
        py_sepred = py_sepred[:min_len]
        cs_ybar = cs_ybar[:min_len]
        cs_sepred = cs_sepred[:min_len]
        print(f"Trimmed to {min_len} points for comparison")

    # Compute metrics
    metrics = compute_metrics(py_ybar, py_sepred, cs_ybar, cs_sepred)

    print(f"\n" + "=" * 70)
    print("RESULTS: Python v23 vs C# Baseline")
    print("=" * 70)
    print(f"  R2 (ybar):        {metrics['R2_ybar']:.6f}  (target >= 0.998)")
    print(f"  R2 (sepred):      {metrics['R2_sepred']:.6f}")
    print(f"  RMSE:             {metrics['RMSE']:.6f}")
    print(f"  MB (Mean Bias):   {metrics['MB']:.6f}  (target ~= 0)")
    print(f"  MaxAE:            {metrics['MaxAE']:.6f}")
    print(f"  MaxAE%:           {metrics['MaxAE_pct']:.6f}%")

    # Check acceptance
    print(f"\n" + "=" * 70)
    print("ACCEPTANCE CHECK")
    print("=" * 70)

    r2_pass = metrics['R2_ybar'] >= 0.998
    mb_pass = abs(metrics['MB']) < 0.5

    print(f"  R2 >= 0.998:      {'PASS' if r2_pass else 'FAIL'} ({metrics['R2_ybar']:.6f})")
    print(f"  MB ~= 0:          {'PASS' if mb_pass else 'FAIL'} ({metrics['MB']:.6f})")

    # Numerical stability check
    print(f"\n" + "=" * 70)
    print("NUMERICAL STABILITY CHECK")
    print("=" * 70)

    if np.any(np.abs(py_ybar) > 1e10):
        print("  FAIL: Predictions contain extreme values (>1e10)")
    else:
        print("  PASS: Predictions are numerically stable")

    if np.any(np.isnan(py_ybar)):
        print("  FAIL: Predictions contain NaN")
    else:
        print("  PASS: No NaN values in predictions")

    if np.any(np.isnan(py_sepred)):
        print("  FAIL: Uncertainties contain NaN")
    else:
        print("  PASS: No NaN values in uncertainties")

    if np.any(py_sepred < 0):
        print("  FAIL: Uncertainties contain negative values")
    else:
        print("  PASS: All uncertainties are non-negative")

    print(f"\n{'=' * 70}")
    print("CONCLUSION")
    print(f"{'=' * 70}")
    print("v23 is a conservative optimization of v20, preserving stability.")
    print(f"v23 R2={metrics['R2_ybar']:.4f}, MB={metrics['MB']:.4f}, RMSE={metrics['RMSE']:.4f}")


if __name__ == "__main__":
    main()