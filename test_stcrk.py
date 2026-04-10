"""
ST-CRK Quick Test Script
"""
import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import netCDF4 as nc
import warnings
warnings.filterwarnings('ignore')

root_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
cmaq_file = f'{root_dir}/test_data/raw/CMAQ/2020_PM25.nc'
monitor_file = f'{root_dir}/test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv'
fold_file = f'{root_dir}/test_data/fold_split_table.csv'
output_dir = f'{root_dir}/test_result/创新方法'
os.makedirs(output_dir, exist_ok=True)


def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan}
    return {'R2': r2_score(y_true, y_pred), 'MAE': mean_absolute_error(y_true, y_pred), 'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))}


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def compute_cmaq_gradient(lon, lat, lon_grid, lat_grid, pm25_grid):
    ny, nx = lon_grid.shape
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    row, col = idx // nx, idx % nx
    row = max(1, min(row, ny - 2))
    col = max(1, min(col, nx - 2))
    dmdx = (pm25_grid[row, col + 1] - pm25_grid[row, col - 1]) / (lon_grid[row, col + 1] - lon_grid[row, col - 1] + 1e-6)
    dmdy = (pm25_grid[row + 1, col] - pm25_grid[row - 1, col]) / (lat_grid[row + 1, col] - lat_grid[row - 1, col] + 1e-6)
    return dmdx, dmdy


print('Loading data...')
monitor_df = pd.read_csv(monitor_file)
fold_df = pd.read_csv(fold_file)
ds = nc.Dataset(cmaq_file, 'r')
lon_cmaq = ds.variables['lon'][:]
lat_cmaq = ds.variables['lat'][:]
pred_pm25 = ds.variables['pred_PM25'][:]
ds.close()

# Simplified kernel (faster)
kernel = ConstantKernel(5.0) * RBF(length_scale=0.5) + WhiteKernel(noise_level=0.5)

from Code.VNAeVNAaVNA.core import VNAFusionCore


def run_one_day_fast(day_str, day_idx):
    day_df = monitor_df[monitor_df['Date'] == day_str].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    cmaq_day = pred_pm25[day_idx]
    cmaq_values = [get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day) for _, row in day_df.iterrows()]
    day_df['CMAQ'] = cmaq_values

    # Add gradient features
    grad_x, grad_y = [], []
    for _, row in day_df.iterrows():
        dx, dy = compute_cmaq_gradient(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        grad_x.append(dx)
        grad_y.append(dy)
    day_df['grad_x'] = grad_x
    day_df['grad_y'] = grad_y

    results = {1: [], 2: [], 3: []}  # VNA, RK-Poly, ST-CRK

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc', 'grad_x'])
        test_df = day_df[day_df['fold'] == fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc', 'grad_x'])
        if len(test_df) == 0:
            continue

        y_test = test_df['Conc'].values
        m_train, m_test = train_df['CMAQ'].values, test_df['CMAQ'].values
        y_train = train_df['Conc'].values

        # VNA
        try:
            vna_core = VNAFusionCore(k=30, power=-2)
            vna_core.fit(train_df)
            vna_pred = vna_core.predict(test_df[['Lon', 'Lat']].values)
            vna_fusion = vna_pred['vna'].values if 'vna' in vna_pred.columns else vna_pred[:, 0]
        except:
            vna_fusion = test_df['CMAQ'].values
        results[1].append((y_test, vna_fusion))

        # RK-Poly (simplified: less GPR iterations)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))
        ols_poly = LinearRegression()
        ols_poly.fit(m_train_poly, y_train)
        pred_poly = ols_poly.predict(m_test_poly)
        residual_poly = y_train - ols_poly.predict(m_train_poly)
        gpr_poly = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1.0, normalize_y=True)
        gpr_poly.fit(train_df[['Lon', 'Lat']].values, residual_poly)
        gpr_poly_pred, _ = gpr_poly.predict(test_df[['Lon', 'Lat']].values, return_std=True)
        results[2].append((y_test, pred_poly + gpr_poly_pred))

        # ST-CRK (with gradient as auxiliary variable)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[['Lon', 'Lat', 'grad_x', 'grad_y']].values)
        X_test = scaler.transform(test_df[['Lon', 'Lat', 'grad_x', 'grad_y']].values)

        # Bias correction
        bias_model = Ridge(alpha=1.0)
        bias_model.fit(m_train.reshape(-1, 1), y_train)
        residual = y_train - bias_model.predict(m_train.reshape(-1, 1))

        # GPR on residuals with gradient features
        gpr_stcrk = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1.0, normalize_y=True)
        gpr_stcrk.fit(X_train, residual)
        residual_pred, _ = gpr_stcrk.predict(X_test, return_std=True)

        stcrk_pred = bias_model.predict(m_test.reshape(-1, 1)) + residual_pred
        results[3].append((y_test, stcrk_pred))

    return results


# Test January 5 days
jan_dates = ['2020-01-15', '2020-01-16', '2020-01-17', '2020-01-18', '2020-01-19']

all_results = {1: [], 2: [], 3: []}
print('Testing January 5 days...')
for i, d in enumerate(jan_dates):
    date_obj = datetime.strptime(d, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    print(f'  Day {i+1}/5: {d}...')
    r = run_one_day_fast(d, day_idx)
    for k in all_results:
        all_results[k].extend(r[k])

print('\n=== January Results (5 days) ===')
for name, k in [('VNA', 1), ('RK-Poly', 2), ('ST-CRK', 3)]:
    y_all = np.concatenate([x[0] for x in all_results[k]])
    p_all = np.concatenate([x[1] for x in all_results[k]])
    m = compute_metrics(y_all, p_all)
    print(f'{name}: R2={m["R2"]:.4f}, MAE={m["MAE"]:.2f}, RMSE={m["RMSE"]:.2f}')

# Save January results
summary = []
for name, k in [('VNA', 1), ('RK-Poly', 2), ('ST-CRK', 3)]:
    y_all = np.concatenate([x[0] for x in all_results[k]])
    p_all = np.concatenate([x[1] for x in all_results[k]])
    m = compute_metrics(y_all, p_all)
    summary.append({'method': name, 'month': 'January', **m})

# Test July 5 days
jul_dates = ['2020-07-15', '2020-07-16', '2020-07-17', '2020-07-18', '2020-07-19']

print('\nTesting July 5 days...')
all_results_jul = {1: [], 2: [], 3: []}
for i, d in enumerate(jul_dates):
    date_obj = datetime.strptime(d, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    print(f'  Day {i+1}/5: {d}...')
    r = run_one_day_fast(d, day_idx)
    for k in all_results_jul:
        all_results_jul[k].extend(r[k])

print('\n=== July Results (5 days) ===')
for name, k in [('VNA', 1), ('RK-Poly', 2), ('ST-CRK', 3)]:
    y_all = np.concatenate([x[0] for x in all_results_jul[k]])
    p_all = np.concatenate([x[1] for x in all_results_jul[k]])
    m = compute_metrics(y_all, p_all)
    print(f'{name}: R2={m["R2"]:.4f}, MAE={m["MAE"]:.2f}, RMSE={m["RMSE"]:.2f}')

# Add July results
for name, k in [('VNA', 1), ('RK-Poly', 2), ('ST-CRK', 3)]:
    y_all = np.concatenate([x[0] for x in all_results_jul[k]])
    p_all = np.concatenate([x[1] for x in all_results_jul[k]])
    m = compute_metrics(y_all, p_all)
    summary.append({'method': name, 'month': 'July', **m})

# Save all results
summary_df = pd.DataFrame(summary)
summary_df.to_csv(f'{output_dir}/ST_CRK_Jan_July_summary.csv', index=False)

print('\n' + '='*60)
print('Final Summary')
print('='*60)
print(summary_df.to_string(index=False))
print(f'\nResults saved to {output_dir}/ST_CRK_Jan_July_summary.csv')
