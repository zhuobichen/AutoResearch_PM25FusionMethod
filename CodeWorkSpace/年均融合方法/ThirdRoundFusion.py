"""
年平均数据高级融合方法 (第三轮)
==============================
基于前两轮结果继续优化

已验证有效的方法:
- CMAQGuidedFusion: R2=0.5756 (最好)
- VarianceWeightedFusion: R2=0.5742

第三轮创新方向:
1. OptimizedCMAQGuided: 优化CMAQGuidedFusion的权重参数
2. MultiScaleFusion: 多尺度空间融合
3. AdaptiveWeightFusion: 自适应权重融合
4. GPRResidualFusion: GPR残差融合
5. TrendSurfaceFusion: 趋势面融合
6. BestEnsembleFusion: 最佳方法集成
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from scipy.interpolate import Rbf
import netCDF4 as nc
from Code.VNAeVNAaVNA.nna_methods import NNA

root_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
cmaq_file = f'{root_dir}/test_data/raw/CMAQ/2020_PM25.nc'
monitor_file = f'{root_dir}/test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv'
fold_file = f'{root_dir}/test_data/fold_split_table.csv'
output_dir = f'{root_dir}/test_result/年平均融合测试'
os.makedirs(output_dir, exist_ok=True)


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


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    """获取站点位置的CMAQ值（最近邻）"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def load_and_prepare_year_avg_data():
    """加载并准备年平均数据"""
    print("=== Loading Year-Average Data ===")

    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    monitor_year_avg = monitor_df.groupby('Site').agg({
        'Conc': 'mean',
        'Lat': 'first',
        'Lon': 'first'
    }).reset_index()
    monitor_year_avg.columns = ['Site', 'Conc', 'Lat', 'Lon']

    print(f"Monitor year-avg: {len(monitor_year_avg)} sites")

    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    cmaq_year_avg = pred_pm25.mean(axis=0)
    print(f"CMAQ year-avg shape: {cmaq_year_avg.shape}")

    cmaq_values = []
    for _, row in monitor_year_avg.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_year_avg)
        cmaq_values.append(val)
    monitor_year_avg['CMAQ'] = cmaq_values

    monitor_year_avg = monitor_year_avg.merge(fold_df, on='Site', how='left')
    monitor_year_avg = monitor_year_avg.dropna(subset=['Lat', 'Lon', 'CMAQ', 'Conc', 'fold'])

    print(f"Final dataset: {len(monitor_year_avg)} sites")

    return monitor_year_avg, lon_cmaq, lat_cmaq, cmaq_year_avg


def run_optimized_cmaq_guided(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    OptimizedCMAQGuided: 优化CMAQGuidedFusion的权重参数
    """
    print("\n=== Optimized CMAQ Guided Fusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # VNA
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(train_df[['Lon', 'Lat']].values, train_df['Conc'].values)
        vna_pred_grid = nn_vna.predict(X_grid_full)

        # aVNA
        train_df = train_df.copy()
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(train_df[['Lon', 'Lat']].values, train_df['bias'].values)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_pred_grid = y_grid_model_full + bias_grid

        # 提取预测
        vna_pred = np.zeros(len(test_df))
        avna_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            vna_pred[i] = vna_pred_grid[idx // nx * nx + idx % nx]
            avna_pred[i] = avna_pred_grid[idx // nx * nx + idx % nx]

        # 优化的动态权重
        obs_mean = train_df['Conc'].mean()
        obs_std = train_df['Conc'].std()
        cmaq_test = test_df['CMAQ'].values

        # 计算z-score偏差
        deviation = (cmaq_test - obs_mean) / obs_std
        # 使用tanh压缩到[0,1]范围
        weight_vna = 0.5 * (1 + np.tanh(-0.5 * deviation))

        ocgf_pred = weight_vna * vna_pred + (1 - weight_vna) * avna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'ocgf': ocgf_pred
        }

        print(f"  Fold {fold_id}: completed")

    ocgf_all = np.concatenate([results[f]['ocgf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, ocgf_all)
    print(f"  Optimized CMAQ Guided: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_multiscale_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    MultiScaleFusion: 多尺度空间融合
    使用不同的k值捕获不同尺度的空间结构
    """
    print("\n=== MultiScale Fusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 多尺度VNA
        scales = [
            {'k': 5, 'power': -2, 'weight': 0.2},   # 局部
            {'k': 15, 'power': -2, 'weight': 0.4},  # 中等
            {'k': 30, 'power': -2, 'weight': 0.4},  # 区域
        ]

        ms_vna_preds = []
        for scale in scales:
            nn = NNA(method='nearest', k=scale['k'], power=scale['power'])
            nn.fit(train_df[['Lon', 'Lat']].values, train_df['Conc'].values)
            pred_grid = nn.predict(X_grid_full)

            pred = np.zeros(len(test_df))
            for i in range(len(test_df)):
                dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
                idx = np.argmin(dist)
                pred[i] = pred_grid[idx // nx * nx + idx % nx]
            ms_vna_preds.append(scale['weight'] * pred)

        ms_vna_pred = np.sum(ms_vna_preds, axis=0)

        # aVNA
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(train_df[['Lon', 'Lat']].values, train_df['bias'].values)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_pred_grid = y_grid_model_full + bias_grid

        avna_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            avna_pred[i] = avna_pred_grid[idx // nx * nx + idx % nx]

        # 融合多尺度VNA和aVNA
        cmaq_test = test_df['CMAQ'].values
        obs_mean = train_df['Conc'].mean()
        deviation = np.abs(cmaq_test - obs_mean) / obs_mean
        weight_vna = np.clip(1 - 0.5 * deviation, 0.3, 0.7)

        msf_pred = weight_vna * ms_vna_pred + (1 - weight_vna) * avna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'msf': msf_pred
        }

        print(f"  Fold {fold_id}: completed")

    msf_all = np.concatenate([results[f]['msf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, msf_all)
    print(f"  MultiScale Fusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_adaptive_weight_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    AdaptiveWeightFusion: 自适应权重融合
    根据训练数据自动学习最优权重组合
    """
    print("\n=== Adaptive Weight Fusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 多种基础预测
        # VNA
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(train_df[['Lon', 'Lat']].values, train_df['Conc'].values)
        vna_pred_grid = nn_vna.predict(X_grid_full)

        # eVNA
        train_df['ratio'] = train_df['Conc'] / train_df['CMAQ']
        nn_ratio = NNA(method='nearest', k=15, power=-2)
        nn_ratio.fit(train_df[['Lon', 'Lat']].values, train_df['ratio'].values)
        ratio_grid = nn_ratio.predict(X_grid_full)
        evna_pred_grid = y_grid_model_full * ratio_grid

        # aVNA
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(train_df[['Lon', 'Lat']].values, train_df['bias'].values)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_pred_grid = y_grid_model_full + bias_grid

        # 提取预测
        vna_pred = np.zeros(len(test_df))
        evna_pred = np.zeros(len(test_df))
        avna_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            vna_pred[i] = vna_pred_grid[idx // nx * nx + idx % nx]
            evna_pred[i] = evna_pred_grid[idx // nx * nx + idx % nx]
            avna_pred[i] = avna_pred_grid[idx // nx * nx + idx % nx]

        # 用训练数据学习最优权重
        vna_train = nn_vna.predict(train_df[['Lon', 'Lat']].values)
        evna_train = train_df['CMAQ'].values * nn_ratio.predict(train_df[['Lon', 'Lat']].values)
        avna_train = train_df['CMAQ'].values + nn_bias.predict(train_df[['Lon', 'Lat']].values)

        # 优化权重
        best_r2 = -999
        best_weights = (1/3, 1/3, 1/3)

        for w1 in np.arange(0.2, 0.8, 0.1):
            for w2 in np.arange(0.1, 0.6, 0.1):
                w3 = 1 - w1 - w2
                if w3 < 0.1:
                    continue
                blend = w1 * vna_train + w2 * evna_train + w3 * avna_train
                r2 = r2_score(train_df['Conc'].values, blend)
                if r2 > best_r2:
                    best_r2 = r2
                    best_weights = (w1, w2, w3)

        # 应用最优权重
        awf_pred = best_weights[0] * vna_pred + best_weights[1] * evna_pred + best_weights[2] * avna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'awf': awf_pred
        }

        print(f"  Fold {fold_id}: completed (weights: VNA={best_weights[0]:.2f}, eVNA={best_weights[1]:.2f}, aVNA={best_weights[2]:.2f})")

    awf_all = np.concatenate([results[f]['awf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, awf_all)
    print(f"  Adaptive Weight Fusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_gpr_residual_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    GPRResidualFusion: GPR残差融合
    使用GPR建模空间残差结构
    """
    print("\n=== GPR Residual Fusion ===")

    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # aVNA预测
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(train_df[['Lon', 'Lat']].values, train_df['bias'].values)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_pred_grid = y_grid_model_full + bias_grid

        # 计算训练数据的aVNA预测
        avna_train = train_df['CMAQ'].values + nn_bias.predict(train_df[['Lon', 'Lat']].values)
        residual_train = train_df['Conc'].values - avna_train

        # GPR建模残差
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1.0, normalize_y=True)
        gpr.fit(train_df[['Lon', 'Lat']].values, residual_train)
        gpr_pred, _ = gpr.predict(test_df[['Lon', 'Lat']].values, return_std=True)

        # 提取aVNA预测
        avna_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            avna_pred[i] = avna_pred_grid[idx // nx * nx + idx % nx]

        grf_pred = avna_pred + gpr_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'grf': grf_pred
        }

        print(f"  Fold {fold_id}: completed")

    grf_all = np.concatenate([results[f]['grf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, grf_all)
    print(f"  GPR Residual Fusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_trend_surface_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    TrendSurfaceFusion: 趋势面融合
    分离趋势（全局）和局部异常（残差）
    """
    print("\n=== Trend Surface Fusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 1. 拟合全局趋势面 (使用CMAQ + 空间坐标)
        X_train = np.column_stack([
            train_df['CMAQ'].values,
            train_df['Lat'].values,
            train_df['Lon'].values,
            train_df['Lat'].values * train_df['Lon'].values,
            train_df['Lat'].values ** 2,
            train_df['Lon'].values ** 2
        ])
        y_train = train_df['Conc'].values

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # 2. 预测网格上的趋势
        X_grid = np.column_stack([
            y_grid_model_full,
            lat_cmaq.ravel(),
            lon_cmaq.ravel(),
            lat_cmaq.ravel() * lon_cmaq.ravel(),
            lat_cmaq.ravel() ** 2,
            lon_cmaq.ravel() ** 2
        ])
        trend_grid = lr.predict(X_grid)

        # 3. 计算训练数据的残差
        trend_train = lr.predict(X_train)
        residual_train = y_train - trend_train

        # 4. 空间插值残差
        nn_res = NNA(method='nearest', k=15, power=-2)
        nn_res.fit(train_df[['Lon', 'Lat']].values, residual_train)
        residual_grid = nn_res.predict(X_grid_full)

        # 5. 融合
        fusion_grid = trend_grid + residual_grid

        # 提取测试预测
        tsf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            tsf_pred[i] = fusion_grid[idx // nx * nx + idx % nx]

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'tsf': tsf_pred
        }

        print(f"  Fold {fold_id}: completed")

    tsf_all = np.concatenate([results[f]['tsf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, tsf_all)
    print(f"  Trend Surface Fusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_best_ensemble_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    BestEnsembleFusion: 最佳方法集成
    集成CMAQGuidedFusion和VarianceWeightedFusion
    """
    print("\n=== Best Ensemble Fusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # VNA
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(train_df[['Lon', 'Lat']].values, train_df['Conc'].values)
        vna_pred_grid = nn_vna.predict(X_grid_full)

        # aVNA
        train_df = train_df.copy()
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(train_df[['Lon', 'Lat']].values, train_df['bias'].values)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_pred_grid = y_grid_model_full + bias_grid

        # eVNA
        train_df['ratio'] = train_df['Conc'] / train_df['CMAQ']
        nn_ratio = NNA(method='nearest', k=15, power=-2)
        nn_ratio.fit(train_df[['Lon', 'Lat']].values, train_df['ratio'].values)
        ratio_grid = nn_ratio.predict(X_grid_full)
        evna_pred_grid = y_grid_model_full * ratio_grid

        # 提取预测
        vna_pred = np.zeros(len(test_df))
        avna_pred = np.zeros(len(test_df))
        evna_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            vna_pred[i] = vna_pred_grid[idx // nx * nx + idx % nx]
            avna_pred[i] = avna_pred_grid[idx // nx * nx + idx % nx]
            evna_pred[i] = evna_pred_grid[idx // nx * nx + idx % nx]

        # CMAQ Guided权重
        obs_mean = train_df['Conc'].mean()
        cmaq_test = test_df['CMAQ'].values
        deviation = np.abs(cmaq_test - obs_mean) / obs_mean
        weight_cgf = np.clip(1 - deviation, 0.3, 0.7)
        cgf_pred = weight_cgf * vna_pred + (1 - weight_cgf) * avna_pred

        # Variance Weighted权重
        cmaq_gradient = np.gradient(cmaq_year_avg, axis=0) ** 2 + np.gradient(cmaq_year_avg, axis=1) ** 2
        cmaq_variance_grid = cmaq_gradient.ravel()
        variance_weight = cmaq_variance_grid / (cmaq_variance_grid.max() + 1e-6)
        vna_weight_vwf = np.clip(0.5 + 0.3 * variance_weight, 0.3, 0.7)
        vwf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            w = vna_weight_vwf[idx // nx * nx + idx % nx]
            vwf_pred[i] = w * vna_pred[i] + (1 - w) * avna_pred[i]

        # 集成CGF和VWF (权重0.6和0.4)
        bef_pred = 0.6 * cgf_pred + 0.4 * vwf_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'bef': bef_pred
        }

        print(f"  Fold {fold_id}: completed")

    bef_all = np.concatenate([results[f]['bef'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, bef_all)
    print(f"  Best Ensemble Fusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("年平均数据高级融合方法测试 (第三轮)")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    methods = [
        ('OptimizedCMAQGuided', run_optimized_cmaq_guided),
        ('MultiScaleFusion', run_multiscale_fusion),
        ('AdaptiveWeightFusion', run_adaptive_weight_fusion),
        ('GPRResidualFusion', run_gpr_residual_fusion),
        ('TrendSurfaceFusion', run_trend_surface_fusion),
        ('BestEnsembleFusion', run_best_ensemble_fusion),
    ]

    for name, func in methods:
        metrics, _ = func(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
        results[name] = metrics

    print("\n" + "="*60)
    print("第三轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/third_round_results.csv', index=False)

    print("\n方法排名 (按R2降序):")
    for _, row in results_df.iterrows():
        print(f"  {row['method']:25s}: R2={row['R2']:.4f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}")

    return results_df


if __name__ == '__main__':
    results_df = main()