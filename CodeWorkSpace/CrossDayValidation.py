"""
Cross-Day Validation Script - Using SuperStackingEnsemble Learned Weights
Test on Day 2 (2020-01-02) and Day 3 (2020-01-03)
Optimized version with faster execution
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import netCDF4 as nc
from Code.VNAeVNAaVNA.nna_methods import NNA

root_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
cmaq_file = root_dir + '/test_data/raw/CMAQ/2020_PM25.nc'
monitor_file = root_dir + '/test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv'
fold_file = root_dir + '/test_data/fold_split_table.csv'
output_dir = root_dir + '/test_result/cross_day_validation'
os.makedirs(output_dir, exist_ok=True)

# SuperStackingEnsemble learned weights (from user task description)
WEIGHTS = {
    'RK_Poly': 2.934,
    'RK_Poly3': -0.939,
    'RK_OLS': -0.996,
    'eVNA': 0.206,
    'aVNA': -0.216,
    'CMAQ': 0.100
}

print("="*70, flush=True)
print("Cross-Day Validation - SuperStackingEnsemble Weights", flush=True)
print("="*70, flush=True)
print("\nUsing weights:", flush=True)
for name, w in WEIGHTS.items():
    print(f"  {name}: {w:+.3f}", flush=True)


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
    """Get CMAQ value at site"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def apply_stacking_weights(predictions, weights):
    """
    Weighted fusion using fixed weights
    predictions: dict with keys 'RK_Poly', 'RK_Poly3', 'RK_OLS', 'eVNA', 'aVNA', 'CMAQ'
    weights: dict with same keys
    """
    result = np.zeros_like(list(predictions.values())[0])
    for name, pred in predictions.items():
        result += weights[name] * pred
    return result


def run_cross_day_validation(selected_day='2020-01-02'):
    """
    Run 10-fold cross validation on specified day
    """
    print("\n" + "="*70, flush=True)
    print(f"Processing Date: {selected_day}", flush=True)
    print("="*70, flush=True)

    # Load data
    print("\n=== Loading Data ===", flush=True)
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    # Load CMAQ data
    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    from datetime import datetime
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    pred_day = pred_pm25[day_idx]

    # Extract CMAQ at sites
    print("=== Extracting CMAQ at Sites ===", flush=True)
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = pred_day.ravel()

    print(f"Data loaded: {len(day_df)} monitoring records", flush=True)

    # Define GPR kernel (faster settings)
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    print("\n=== Running 10-Fold Cross Validation ===", flush=True)
    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        print(f"\n--- Fold {fold_id} ---", flush=True)
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

        # === 1. RK-Poly ===
        print(f"  RK-Poly...", flush=True)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols_poly = LinearRegression()
        ols_poly.fit(m_train_poly, y_train)
        pred_poly = ols_poly.predict(m_test_poly)
        residual_poly = y_train - ols_poly.predict(m_train_poly)

        gpr_poly = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=0.1, normalize_y=True)
        gpr_poly.fit(X_train, residual_poly)
        gpr_poly_pred, _ = gpr_poly.predict(X_test, return_std=True)
        rk_poly_pred = pred_poly + gpr_poly_pred

        # === 2. RK-Poly3 ===
        print(f"  RK-Poly3...", flush=True)
        poly3 = PolynomialFeatures(degree=3, include_bias=False)
        m_train_poly3 = poly3.fit_transform(m_train.reshape(-1, 1))
        m_test_poly3 = poly3.transform(m_test.reshape(-1, 1))

        ols_poly3 = LinearRegression()
        ols_poly3.fit(m_train_poly3, y_train)
        pred_poly3 = ols_poly3.predict(m_test_poly3)
        residual_poly3 = y_train - ols_poly3.predict(m_train_poly3)

        gpr_poly3 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=0.1, normalize_y=True)
        gpr_poly3.fit(X_train, residual_poly3)
        gpr_poly3_pred, _ = gpr_poly3.predict(X_test, return_std=True)
        rk_poly3_pred = pred_poly3 + gpr_poly3_pred

        # === 3. RK-OLS ===
        print(f"  RK-OLS...", flush=True)
        ols = LinearRegression()
        ols.fit(m_train.reshape(-1, 1), y_train)
        pred_ols = ols.predict(m_test.reshape(-1, 1))
        residual_ols = y_train - ols.predict(m_train.reshape(-1, 1))

        gpr_ols = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=0.1, normalize_y=True)
        gpr_ols.fit(X_train, residual_ols)
        gpr_ols_pred, _ = gpr_ols.predict(X_test, return_std=True)
        rk_ols_pred = pred_ols + gpr_ols_pred

        # === 4. eVNA ===
        print(f"  eVNA...", flush=True)
        train_df = train_df.copy()
        train_df['x'] = train_df['Lon']
        train_df['y'] = train_df['Lat']
        train_df['mod'] = train_df['CMAQ']
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        train_df['rn'] = train_df['Conc'] / train_df['CMAQ']

        nn = NNA(method='voronoi', k=30, power=-2)
        nn.fit(train_df[['x', 'y']], train_df[['bias', 'rn']])

        zdf_grid = nn.predict(X_grid_full, njobs=4)
        bias_grid = zdf_grid[:, 0]
        rn_grid = zdf_grid[:, 1]

        evna_pred = np.zeros(len(test_df))
        for i, (_, row) in enumerate(test_df.iterrows()):
            dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
            idx = np.argmin(dist)
            evna_pred[i] = y_grid_model_full[idx] * rn_grid[idx]

        # === 5. aVNA ===
        print(f"  aVNA...", flush=True)
        avna_pred = np.zeros(len(test_df))
        for i, (_, row) in enumerate(test_df.iterrows()):
            dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
            idx = np.argmin(dist)
            avna_pred[i] = m_test[i] + bias_grid[idx]

        results[fold_id] = {
            'y_true': y_test,
            'm_test': m_test,
            'rk_poly': rk_poly_pred,
            'rk_poly3': rk_poly3_pred,
            'rk_ols': rk_ols_pred,
            'evna': evna_pred,
            'avna': avna_pred
        }

        print(f"  Fold {fold_id}: completed", flush=True)

    # Aggregate all fold results
    rk_poly_all = np.concatenate([results[f]['rk_poly'] for f in range(1, 11) if results[f]])
    rk_poly3_all = np.concatenate([results[f]['rk_poly3'] for f in range(1, 11) if results[f]])
    rk_ols_all = np.concatenate([results[f]['rk_ols'] for f in range(1, 11) if results[f]])
    evna_all = np.concatenate([results[f]['evna'] for f in range(1, 11) if results[f]])
    avna_all = np.concatenate([results[f]['avna'] for f in range(1, 11) if results[f]])
    m_test_all = np.concatenate([results[f]['m_test'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # Compute individual method R2
    print("\n=== Individual Method R2 ===", flush=True)
    individual_metrics = {
        'RK-Poly': compute_metrics(true_all, rk_poly_all),
        'RK-Poly3': compute_metrics(true_all, rk_poly3_all),
        'RK-OLS': compute_metrics(true_all, rk_ols_all),
        'eVNA': compute_metrics(true_all, evna_all),
        'aVNA': compute_metrics(true_all, avna_all),
        'CMAQ': compute_metrics(true_all, m_test_all)
    }
    for name, metrics in individual_metrics.items():
        print(f"  {name}: R2={metrics['R2']:.4f}", flush=True)

    # Use fixed weights to compute SuperStackingEnsemble
    print("\n=== SuperStackingEnsemble (Fixed Weights) ===", flush=True)
    predictions = {
        'RK_Poly': rk_poly_all,
        'RK_Poly3': rk_poly3_all,
        'RK_OLS': rk_ols_all,
        'eVNA': evna_all,
        'aVNA': avna_all,
        'CMAQ': m_test_all
    }
    stacked_pred = apply_stacking_weights(predictions, WEIGHTS)
    stacked_metrics = compute_metrics(true_all, stacked_pred)

    print(f"\n  Ensemble Result:", flush=True)
    print(f"    R2: {stacked_metrics['R2']:.4f}", flush=True)
    print(f"    MAE: {stacked_metrics['MAE']:.2f}", flush=True)
    print(f"    RMSE: {stacked_metrics['RMSE']:.2f}", flush=True)
    print(f"    MB: {stacked_metrics['MB']:.2f}", flush=True)

    return {
        'day': selected_day,
        'n_samples': len(true_all),
        'individual': individual_metrics,
        'stacked': stacked_metrics
    }


def main():
    """Main function"""
    print("\n" + "="*70, flush=True)
    print("Cross-Day Validation Started", flush=True)
    print("="*70, flush=True)

    # Run validation on Day 1, Day 2, Day 3
    days = ['2020-01-01', '2020-01-02', '2020-01-03']
    all_results = {}

    for day in days:
        try:
            result = run_cross_day_validation(day)
            all_results[day] = result
        except Exception as e:
            print(f"\nError processing {day}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            traceback.print_exc(file=sys.stdout)
            flush=True

    # Generate comparison table
    print("\n" + "="*70, flush=True)
    print("Cross-Day Validation Results Summary", flush=True)
    print("="*70, flush=True)

    # Create comparison CSV
    rows = []
    for day, result in all_results.items():
        # Individual methods
        for method, metrics in result['individual'].items():
            rows.append({
                'Date': day,
                'Method': method,
                'R2': metrics['R2'],
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'MB': metrics['MB']
            })
        # Ensemble method
        rows.append({
            'Date': day,
            'Method': 'SuperStackingEnsemble',
            'R2': result['stacked']['R2'],
            'MAE': result['stacked']['MAE'],
            'RMSE': result['stacked']['RMSE'],
            'MB': result['stacked']['MB']
        })

    results_df = pd.DataFrame(rows)
    results_csv_path = output_dir + '/cross_day_validation_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to: {results_csv_path}", flush=True)

    # Generate report
    report = f"""# Cross-Day Validation Report - SuperStackingEnsemble Weights

## Weights Used (from SuperStackingEnsemble learned on Day 1)

| Model | Weight |
|-------|--------|
| RK-Poly | {WEIGHTS['RK_Poly']:+.3f} |
| RK-Poly3 | {WEIGHTS['RK_Poly3']:+.3f} |
| RK-OLS | {WEIGHTS['RK_OLS']:+.3f} |
| eVNA | {WEIGHTS['eVNA']:+.3f} |
| aVNA | {WEIGHTS['aVNA']:+.3f} |
| CMAQ | {WEIGHTS['CMAQ']:+.3f} |

## Results Comparison by Date

### Individual Method R2 Comparison

| Method | Day 1 | Day 2 | Day 3 |
|--------|-------|-------|-------|
"""

    # Get R2 for each method across days
    methods = ['RK-Poly', 'RK-Poly3', 'RK-OLS', 'eVNA', 'aVNA', 'CMAQ', 'SuperStackingEnsemble']
    for method in methods:
        row = f"| {method} |"
        for day in days:
            if day in all_results:
                if method == 'SuperStackingEnsemble':
                    r2 = all_results[day]['stacked']['R2']
                else:
                    r2 = all_results[day]['individual'][method]['R2']
                row += f" {r2:.4f} |"
            else:
                row += " N/A |"
        report += row + "\n"

    report += f"""
### Detailed Metrics Comparison

#### SuperStackingEnsemble

| Date | Samples | R2 | MAE | RMSE | MB |
|------|---------|-----|-----|------|-----|
"""

    for day in days:
        if day in all_results:
            m = all_results[day]['stacked']
            report += f"| {day} | {all_results[day]['n_samples']} | {m['R2']:.4f} | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['MB']:.2f} |\n"

    report += f"""
### Analysis

1. **Cross-Day Generalization**: Using weights learned from Day 1, test on Day 2 and Day 3 data to observe if the weights have cross-day applicability.

2. **Weight Interpretation**:
   - RK-Poly weight is {WEIGHTS['RK_Poly']:+.3f}, the main contributing model
   - RK-Poly3 ({WEIGHTS['RK_Poly3']:+.3f}) and RK-OLS ({WEIGHTS['RK_OLS']:+.3f}) have negative weights, serving as correction factors
   - eVNA ({WEIGHTS['eVNA']:+.3f}) and aVNA ({WEIGHTS['aVNA']:+.3f}) provide spatial correction
   - CMAQ ({WEIGHTS['CMAQ']:+.3f}) serves as the base reference

3. **Comparison with Day 1**: Observe the difference in performance of each method on Day 2 and Day 3 compared to Day 1 to evaluate model stability.
"""

    report_path = output_dir + '/cross_day_validation_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to: {report_path}", flush=True)

    # Print table
    print("\n" + "="*70, flush=True)
    print("SuperStackingEnsemble R2 Comparison Table", flush=True)
    print("="*70, flush=True)
    print(f"{'Method':<25} {'Day 1':>10} {'Day 2':>10} {'Day 3':>10}", flush=True)
    print("-"*55, flush=True)
    for method in methods:
        row = f"{method:<25}"
        for day in days:
            if day in all_results:
                if method == 'SuperStackingEnsemble':
                    r2 = all_results[day]['stacked']['R2']
                else:
                    r2 = all_results[day]['individual'][method]['R2']
                row += f" {r2:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        print(row, flush=True)

    return all_results


if __name__ == '__main__':
    results = main()
