"""
PM2.5 Fusion Method 10-Fold Cross Validation
=============================================
Validation Steps:
1. Run VNA/eVNA/aVNA baseline methods, confirm R2 in reasonable range
2. Perform 10-fold cross-validation for each method
3. Innovation validation: compare with baseline
4. Output results to test_result/InnovationMethods/
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from datetime import datetime
import netCDF4 as nc
import warnings
warnings.filterwarnings('ignore')

# Fix UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

root_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
sys.path.insert(0, f'{root_dir}/Code')
sys.path.insert(0, f'{root_dir}/Code/VNAeVNAaVNA')
sys.path.insert(0, f'{root_dir}/Code/VNAeVNAaVNA/nna_methods')

from VNAeVNAaVNA.core import VNAFusionCore
from nna_methods import NNA

output_dir = f'{root_dir}/test_result/InnovationMethods'
os.makedirs(output_dir, exist_ok=True)


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MB': np.mean(y_pred - y_true)
    }


def load_data(selected_day='2020-01-01'):
    """Load test data"""
    print(f"=== Loading data: {selected_day} ===")

    # Load monitor data
    monitor_df = pd.read_csv(f'{root_dir}/test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
    fold_df = pd.read_csv(f'{root_dir}/test_data/fold_split_table.csv')

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')

    # Load CMAQ data
    ds = nc.Dataset(f'{root_dir}/test_data/raw/CMAQ/2020_PM25.nc', 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    pred_day = pred_pm25[day_idx]

    def get_cmaq_at_site(lon, lat):
        dist = np.sqrt((lon_cmaq - lon)**2 + (lat_cmaq - lat)**2)
        idx = np.argmin(dist)
        ny, nx = lon_cmaq.shape
        row, col = idx // nx, idx % nx
        return pred_day[row, col]

    day_df['CMAQ'] = day_df.apply(lambda r: get_cmaq_at_site(r['Lon'], r['Lat']), axis=1)

    # Filter valid data
    day_df = day_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc', 'fold'])

    print(f"Total data: {len(day_df)} records")

    # Grid data
    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_model_grid_full = pred_day.ravel()

    return day_df, X_grid_full, y_model_grid_full, lon_cmaq, lat_cmaq


class VNAMethod:
    """VNA/eVNA/aVNA unified interface"""
    def __init__(self, method_type='vna'):
        self.method_type = method_type
        self.core = None

    def fit(self, X_train, y_train, y_model_train, train_df):
        """Train model"""
        obs_df = pd.DataFrame({
            'x': X_train[:, 0],
            'y': X_train[:, 1],
            'Conc': y_train,
            'mod': y_model_train,
            'bias': y_model_train - y_train,
            'r_n': y_train / (y_model_train + 1e-8)
        })

        self.core = VNAFusionCore(k=30, power=-2, method='voronoi')
        self.core.fit(obs_df)
        return self

    def predict(self, X_test_coords, y_model_grid_full, lon_cmaq, lat_cmaq, test_df):
        """Predict"""
        # Predict to grid
        result = self.core.predict(X_test_coords)

        # Select output based on method type
        if self.method_type == 'vna':
            pred_grid = result['vna'].values
        elif self.method_type == 'evna':
            # eVNA = model * r_n
            pred_grid = self.core.compute_evna(y_model_grid_full, result['vna_rn'].values)
        elif self.method_type == 'avna':
            # aVNA = model + bias
            pred_grid = self.core.compute_avna(y_model_grid_full, result['vna_bias'].values)
        else:
            pred_grid = result['vna'].values

        # Interpolate to test points
        y_pred = np.zeros(len(test_df))
        for i, (_, row) in enumerate(test_df.iterrows()):
            dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
            idx = np.argmin(dist)
            y_pred[i] = pred_grid[idx]

        return y_pred


def ten_fold_cv_vna(method, day_df, X_grid_full, y_model_grid_full, lon_cmaq, lat_cmaq):
    """Perform 10-fold CV for VNA/eVNA/aVNA"""
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(day_df), 1):
        train_df = day_df.iloc[train_idx]
        test_df = day_df.iloc[test_idx]

        X_train = train_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values
        y_model_train = train_df['CMAQ'].values

        # Train
        method.fit(X_train, y_train, y_model_train, train_df)

        # Predict
        y_pred = method.predict(X_grid_full, y_model_grid_full, lon_cmaq, lat_cmaq, test_df)

        # Compute metrics
        y_test = test_df['Conc'].values
        m = compute_metrics(y_test, y_pred)
        m['fold'] = fold_id
        metrics_list.append(m)

        print(f"  Fold {fold_id}: R2={m['R2']:.4f}, MAE={m['MAE']:.2f}, RMSE={m['RMSE']:.2f}")

    # Aggregate
    avg_metrics = {
        'R2': np.mean([m['R2'] for m in metrics_list]),
        'MAE': np.mean([m['MAE'] for m in metrics_list]),
        'RMSE': np.mean([m['RMSE'] for m in metrics_list]),
        'MB': np.mean([m['MB'] for m in metrics_list])
    }
    return avg_metrics, metrics_list


def main():
    print("=" * 70)
    print("PM2.5 Fusion Method 10-Fold Cross Validation")
    print("=" * 70)

    # Load data
    day_df, X_grid_full, y_model_grid_full, lon_cmaq, lat_cmaq = load_data('2020-01-01')

    # ============================================================
    # Step 1: Baseline Validation
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 1: Baseline Validation (VNA/eVNA/aVNA)")
    print("=" * 70)

    baseline_results = {}

    for method_type in ['vna', 'evna', 'avna']:
        print(f"\n--- {method_type.upper()} ---")
        method = VNAMethod(method_type=method_type)
        avg_metrics, fold_metrics = ten_fold_cv_vna(method, day_df, X_grid_full, y_model_grid_full, lon_cmaq, lat_cmaq)

        baseline_results[method_type] = avg_metrics
        print(f"  Avg: R2={avg_metrics['R2']:.4f}, MAE={avg_metrics['MAE']:.2f}, RMSE={avg_metrics['RMSE']:.2f}, MB={avg_metrics['MB']:.2f}")

        # Check if R2 is in reasonable range
        r2 = avg_metrics['R2']
        expected_range = {'vna': (0.70, 0.95), 'evna': (0.65, 0.95), 'avna': (0.65, 0.95)}
        r_min, r_max = expected_range[method_type]
        if r2 < r_min:
            print(f"  [WARN] R2={r2:.4f} below expected range [{r_min}, {r_max}]")
        elif r2 > r_max:
            print(f"  [WARN] R2={r2:.4f} above expected range [{r_min}, {r_max}]")
        else:
            print(f"  [PASS] R2 in expected range")

    # Save baseline results
    baseline_df = pd.DataFrame([
        {'method': k.upper(), 'R2': v['R2'], 'MAE': v['MAE'], 'RMSE': v['RMSE'], 'MB': v['MB']}
        for k, v in baseline_results.items()
    ])
    baseline_df.to_csv(f'{output_dir}/benchmark_vna_evna_avna_summary.csv', index=False)
    print(f"\nBaseline saved to: {output_dir}/benchmark_vna_evna_avna_summary.csv")

    # Determine best baseline
    best_baseline = max(baseline_results.items(), key=lambda x: x[1]['R2'])
    best_method_name = best_baseline[0].upper()
    best_r2 = best_baseline[1]['R2']
    best_rmse = best_baseline[1]['RMSE']
    best_mb = best_baseline[1]['MB']

    print(f"\nBest baseline: {best_method_name}, R2={best_r2:.4f}, RMSE={best_rmse:.2f}, MB={best_mb:.2f}")

    # ============================================================
    # Step 2: Innovation Validation
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 2: Innovation Validation")
    print("=" * 70)

    # Also check the original Chinese-named directory
    chinese_output_dir = f'{root_dir}/test_result/创新方法'
    all_innovation_files = []

    for check_dir in [output_dir, chinese_output_dir]:
        if os.path.exists(check_dir):
            files = [f for f in os.listdir(check_dir) if f.endswith('_summary.csv') and not f.startswith('benchmark')]
            all_innovation_files.extend([(check_dir, f) for f in files])

    # Remove duplicates
    seen = set()
    unique_files = []
    for dir_path, file_name in all_innovation_files:
        if file_name not in seen:
            seen.add(file_name)
            unique_files.append((dir_path, file_name))

    if unique_files:
        print(f"\nFound {len(unique_files)} innovation method results")

        innovation_summary = []
        for dir_path, csv_file in unique_files:
            method_name = csv_file.replace('_summary.csv', '')
            df = pd.read_csv(f'{dir_path}/{csv_file}')

            if 'R2' in df.columns:
                r2 = df['R2'].iloc[0] if len(df) > 0 else np.nan
                mae = df['MAE'].iloc[0] if 'MAE' in df.columns else np.nan
                rmse = df['RMSE'].iloc[0] if 'RMSE' in df.columns else np.nan
                mb = df['MB'].iloc[0] if 'MB' in df.columns else np.nan

                innovation_summary.append({
                    'method': method_name,
                    'R2': r2,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MB': mb
                })

        if innovation_summary:
            innov_df = pd.DataFrame(innovation_summary)
            print("\nInnovation methods summary:")
            print(innov_df.to_string(index=False))

            # Validate innovation
            print("\n" + "=" * 70)
            print("Innovation validation criteria:")
            print(f"  Best baseline R2: {best_r2:.4f} ({best_method_name})")
            print(f"  Innovation requires: R2 >= best + 0.01, RMSE <= best, |MB| <= best")
            print("=" * 70)

            for _, row in innov_df.iterrows():
                method = row['method']
                r2 = row['R2']
                rmse = row['RMSE']
                mb = abs(row['MB'])

                r2_ok = r2 >= best_r2 + 0.01
                rmse_ok = rmse <= best_rmse
                mb_ok = mb <= abs(best_mb)

                if r2_ok and rmse_ok and mb_ok:
                    status = "[PASS] Innovation validated"
                elif r2 >= best_r2:
                    status = "[PARTIAL] R2 improved but other metrics not met"
                else:
                    status = "[FAIL] Does not exceed baseline"

                print(f"\n{method}:")
                print(f"  R2={r2:.4f} (required>={best_r2+0.01:.4f}, {'OK' if r2_ok else 'FAIL'})")
                print(f"  RMSE={rmse:.2f} (required<={best_rmse:.2f}, {'OK' if rmse_ok else 'FAIL'})")
                print(f"  |MB|={mb:.2f} (required<={abs(best_mb):.2f}, {'OK' if mb_ok else 'FAIL'})")
                print(f"  Status: {status}")
    else:
        print("No innovation method results found")

    print("\n" + "=" * 70)
    print("Validation Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()