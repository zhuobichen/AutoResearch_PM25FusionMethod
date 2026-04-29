"""生成按日期分开的十折验证详细结果CSV"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

root_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
cmaq_file = f'{root_dir}/test_data/raw/CMAQ/2020_PM25.nc'
monitor_file = f'{root_dir}/test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv'
fold_file = f'{root_dir}/test_data/fold_split_table.csv'

def compute_metrics(y_true, y_pred):
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

def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]

def spinkr_predict(X_train, y_train, m_train, X_test, m_test, k=30, spatial_range=0.5):
    bias = y_train - m_train
    sill = np.var(bias)
    dists_train = cdist(X_train, X_train, 'euclidean')
    idx_i, idx_j = np.triu_indices(len(X_train))
    h = dists_train[idx_i, idx_j]
    gamma_exp = 0.5 * (bias[idx_i] - bias[idx_j]) ** 2
    mask = h > 1e-6
    range_est = np.median(h[mask]) if np.any(mask) else spatial_range
    if range_est < 0.1:
        range_est = spatial_range
    bias_pred = np.zeros(len(X_test))
    for i, x0 in enumerate(X_test):
        dists = cdist([x0], X_train, 'euclidean').ravel()
        idx = np.argpartition(dists, min(k, len(dists)))[:min(k, len(dists))]
        C = sill * np.exp(-3 * (dists[idx] / range_est) ** 2)
        weights = C / (np.sum(C) + 1e-10)
        bias_pred[i] = np.sum(weights * bias[idx])
    return m_test + bias_pred

def bmsf_predict(X_train, y_train, m_train, X_test, m_test, k=30, sigma_model=5.0, sigma_obs=2.0, corr_scale=0.5):
    bias = y_train - m_train
    mu_prior, sigma_prior = np.mean(bias), np.std(bias)
    pred = np.zeros(len(X_test))
    for i, x0 in enumerate(X_test):
        dists = cdist([x0], X_train, 'euclidean').ravel()
        idx = np.argpartition(dists, min(k, len(dists)))[:min(k, len(dists))]
        C = np.exp(-0.5 * (dists[idx] / corr_scale) ** 2)
        weights = C / (np.sum(C) + 1e-10)
        local_bias = np.sum(weights * bias[idx])
        w_local = sigma_model ** 2 / (sigma_model ** 2 + sigma_obs ** 2)
        w_prior = sigma_obs ** 2 / (sigma_model ** 2 + sigma_obs ** 2)
        post_bias = w_local * local_bias + w_prior * mu_prior
        pred[i] = m_test[i] + post_bias
    return pred

def corriff_predict(X_train, y_train, m_train, X_test, m_test, k=30, diffusion_coef=0.3, corr_scale=0.5):
    bias = y_train - m_train
    pred = np.zeros(len(X_test))
    for i, x0 in enumerate(X_test):
        dists = cdist([x0], X_train, 'euclidean').ravel()
        idx = np.argpartition(dists, min(k, len(dists)))[:min(k, len(dists))]
        kernel = np.exp(-0.5 * (dists[idx] / corr_scale) ** 2)
        weights = kernel / (np.sum(kernel) + 1e-10)
        pred[i] = m_test[i] + np.sum(weights * bias[idx])
    return pred

# 加载数据
monitor_df = pd.read_csv(monitor_file)
fold_df = pd.read_csv(fold_file)
ds = nc.Dataset(cmaq_file, 'r')
lon_cmaq = ds.variables['lon'][:]
lat_cmaq = ds.variables['lat'][:]
pred_pm25 = ds.variables['pred_PM25'][:]
ds.close()

selected_days = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']
methods = ['SPIN-Kr', 'BMSF-Geostat', 'CorrDiff-3km', 'RK-Poly', 'CMAQ']

# 按日期存储每折结果
results_by_day = {day: {m: [] for m in methods} for day in selected_days}

for day_str in selected_days:
    day_df = monitor_df[monitor_df['Date'] == day_str].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    date_obj = datetime.strptime(day_str, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    cmaq_day = pred_pm25[day_idx]

    cmaq_values = []
    for _, row in day_df.iterrows():
        cmaq_values.append(get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day))
    day_df['CMAQ'] = cmaq_values

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = day_df[day_df['fold'] == fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0 or len(train_df) == 0:
            continue

        y_test = test_df['Conc'].values
        m_test = test_df['CMAQ'].values

        # CMAQ
        results_by_day[day_str]['CMAQ'].append((y_test, m_test))

        # RK-Poly
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train = train_df['CMAQ'].values
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))
        ols_poly = LinearRegression()
        ols_poly.fit(m_train_poly, train_df['Conc'].values)
        pred_poly = ols_poly.predict(m_test_poly)
        residual_poly = train_df['Conc'].values - ols_poly.predict(m_train_poly)
        kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        gpr_poly = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_poly.fit(train_df[['Lon', 'Lat']].values, residual_poly)
        gpr_poly_pred, _ = gpr_poly.predict(test_df[['Lon', 'Lat']].values, return_std=True)
        results_by_day[day_str]['RK-Poly'].append((y_test, pred_poly + gpr_poly_pred))

        # SPIN-Kr
        spin_pred = spinkr_predict(train_df[['Lon', 'Lat']].values, train_df['Conc'].values, train_df['CMAQ'].values, test_df[['Lon', 'Lat']].values, m_test)
        results_by_day[day_str]['SPIN-Kr'].append((y_test, spin_pred))

        # BMSF-Geostat
        bmsf_pred = bmsf_predict(train_df[['Lon', 'Lat']].values, train_df['Conc'].values, train_df['CMAQ'].values, test_df[['Lon', 'Lat']].values, m_test)
        results_by_day[day_str]['BMSF-Geostat'].append((y_test, bmsf_pred))

        # CorrDiff-3km
        corriff_pred = corriff_predict(train_df[['Lon', 'Lat']].values, train_df['Conc'].values, train_df['CMAQ'].values, test_df[['Lon', 'Lat']].values, m_test)
        results_by_day[day_str]['CorrDiff-3km'].append((y_test, corriff_pred))

# 生成按日期的详细CSV
rows = []
for day_str in selected_days:
    row = {'Date': day_str}
    for method in methods:
        y_all = np.concatenate([r[0] for r in results_by_day[day_str][method]])
        p_all = np.concatenate([r[1] for r in results_by_day[day_str][method]])
        metrics = compute_metrics(y_all, p_all)
        row[f'{method}_R2'] = round(metrics['R2'], 4)
        row[f'{method}_MAE'] = round(metrics['MAE'], 2)
        row[f'{method}_RMSE'] = round(metrics['RMSE'], 2)
        row[f'{method}_MB'] = round(metrics['MB'], 2)
    rows.append(row)

df = pd.DataFrame(rows)
output_file = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/test_result/创新方法/SPINKr_BMSFGeostat_CorrDiff3km_daily_results.csv'
df.to_csv(output_file, index=False)
print(f'CSV已保存: {output_file}')
print()
print(df.to_string(index=False))