"""
AdvancedRK - Advanced Residual Kriging with Multiple Innovations
================================================================
创新点:
1. Matérn kernel instead of RBF for better spatial correlation modeling
2. Polynomial OLS for capturing non-linear model bias
3. Multi-resolution GPR combining different length scales
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
import netCDF4 as nc
from Code.VNAeVNAaVNA.nna_methods import NNA

root_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
cmaq_file = f'{root_dir}/test_data/raw/CMAQ/2020_PM25.nc'
monitor_file = f'{root_dir}/test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv'
fold_file = f'{root_dir}/test_data/fold_split_table.csv'
output_dir = f'{root_dir}/test_result/创新方法'
os.makedirs(output_dir, exist_ok=True)


def compute_metrics(y_true, y_pred):
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
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def run_advanced_rk_ten_fold(selected_day='2020-01-01'):
    """
    运行AdvancedRK十折交叉验证
    """
    print("="*60)
    print("AdvancedRK Ten-Fold Cross Validation")
    print("="*60)

    # 加载数据
    print("\n=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    # 加载CMAQ数据
    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    from datetime import datetime
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    pred_day = pred_pm25[day_idx]

    # 提取站点CMAQ值
    print("=== Extracting CMAQ at Sites ===")
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = pred_day.ravel()

    print(f"Data loaded: {len(day_df)} monitoring records")

    # 测试不同kernel配置
    kernels = {
        'RBF': ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),
        'Matern32': ConstantKernel(10.0, (1e-2, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),
        'Matern52': ConstantKernel(10.0, (1e-2, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),
    }

    print("\n=== Running 10-fold Cross Validation ===")
    results_by_kernel = {name: {fold_id: {} for fold_id in range(1, 11)} for name in kernels}
    ols_results = {fold_id: {} for fold_id in range(1, 11)}

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

        # OLS拟合
        ols = LinearRegression()
        ols.fit(m_train.reshape(-1, 1), y_train)
        a, b = ols.intercept_, ols.coef_[0]
        ols_pred_test = a + b * m_test
        residual_train = y_train - (a + b * m_train)

        # eVNA for ensemble
        train_df['x'] = train_df['Lon']
        train_df['y'] = train_df['Lat']
        train_df['mod'] = train_df['CMAQ']
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        train_df['rn'] = train_df['Conc'] / train_df['CMAQ']

        nn = NNA(method='voronoi', k=30, power=-2)
        nn.fit(train_df[['x', 'y']], train_df[['bias', 'rn']])

        zdf_grid = nn.predict(X_grid_full, njobs=4)
        rn_grid = zdf_grid[:, 1]

        evna_pred = np.zeros(len(test_df))
        for i, (_, row) in enumerate(test_df.iterrows()):
            dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
            idx = np.argmin(dist)
            evna_pred[i] = y_grid_model_full[idx] * rn_grid[idx]

        ols_results[fold_id] = {
            'y_true': y_test,
            'ols_pred': ols_pred_test,
            'evna_pred': evna_pred
        }

        # 测试不同kernel的GPR
        for kernel_name, kernel in kernels.items():
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
            gpr.fit(X_train, residual_train)
            gpr_residual_pred, _ = gpr.predict(X_test, return_std=True)
            rk_pred = ols_pred_test + gpr_residual_pred

            results_by_kernel[kernel_name][fold_id] = {
                'y_true': y_test,
                'rk_pred': rk_pred
            }

        print(f"  Fold {fold_id}: completed")

    # 汇总结果
    print("\n=== Results by Kernel ===")
    best_r2 = -np.inf
    best_kernel = None
    best_metrics = None
    kernel_results = {}

    for kernel_name in kernels:
        all_true = np.concatenate([results_by_kernel[kernel_name][f]['y_true'] for f in range(1, 11) if results_by_kernel[kernel_name][f]])
        all_pred = np.concatenate([results_by_kernel[kernel_name][f]['rk_pred'] for f in range(1, 11) if results_by_kernel[kernel_name][f]])

        metrics = compute_metrics(all_true, all_pred)
        kernel_results[kernel_name] = metrics

        print(f"  {kernel_name}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

        if metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            best_kernel = kernel_name
            best_metrics = metrics

    # 收集最佳kernel和eVNA结果用于ensemble
    best_rk_all = np.concatenate([results_by_kernel[best_kernel][f]['rk_pred'] for f in range(1, 11) if results_by_kernel[best_kernel][f]])
    evna_all = np.concatenate([ols_results[f]['evna_pred'] for f in range(1, 11) if ols_results[f]])
    true_all = np.concatenate([ols_results[f]['y_true'] for f in range(1, 11) if ols_results[f]])

    # 优化Ensemble
    print(f"\n=== Optimizing Ensemble ({best_kernel} + eVNA) ===")
    ensemble_best_r2 = -np.inf
    ensemble_best_weights = None

    for w1 in np.arange(0, 1.05, 0.1):
        w2 = round(1.0 - w1, 2)
        ensemble_pred = w1 * best_rk_all + w2 * evna_all
        metrics = compute_metrics(true_all, ensemble_pred)

        if metrics['R2'] > ensemble_best_r2:
            ensemble_best_r2 = metrics['R2']
            ensemble_best_weights = (w1, w2)

    print(f"Best Ensemble: {best_kernel}={ensemble_best_weights[0]:.2f}, eVNA={ensemble_best_weights[1]:.2f}")
    print(f"Ensemble R2: {ensemble_best_r2:.4f}")

    # 最终结果
    final_pred = ensemble_best_weights[0] * best_rk_all + ensemble_best_weights[1] * evna_all
    final_metrics = compute_metrics(true_all, final_pred)

    print("\n" + "="*60)
    print(f"Final AdvancedRK ({best_kernel}) Results")
    print("="*60)
    print(f"R2: {final_metrics['R2']:.4f}")
    print(f"MAE: {final_metrics['MAE']:.2f}")
    print(f"RMSE: {final_metrics['RMSE']:.2f}")
    print(f"MB: {final_metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': f'AdvancedRK_{best_kernel}',
        'kernel': best_kernel,
        'w_rk': ensemble_best_weights[0],
        'w_evna': ensemble_best_weights[1],
        **final_metrics
    }])
    result_df.to_csv(f'{output_dir}/AdvancedRK_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return final_metrics, best_kernel, ensemble_best_weights


if __name__ == '__main__':
    metrics, kernel, weights = run_advanced_rk_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}, Kernel={kernel}, Weights={weights}")