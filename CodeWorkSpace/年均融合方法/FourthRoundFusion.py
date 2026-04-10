"""
年平均数据高级融合方法 (第四轮)
==============================
基于前三轮分析，继续优化

最佳方法:
- CMAQGuidedFusion: R2=0.5756
- VarianceWeightedFusion: R2=0.5742

第四轮优化方向:
1. 精细化CMAQGuided参数搜索
2. 三方法集成 (VNA, aVNA, eVNA)
3. 混合权重优化
4. 基于距离的局部校正
5. 分位数校正融合
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
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


def extract_grid_pred(test_df, lon_cmaq, lat_cmaq, pred_grid, ny, nx):
    """从网格提取测试点预测"""
    pred = np.zeros(len(test_df))
    for i in range(len(test_df)):
        dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
        idx = np.argmin(dist)
        row_idx, col_idx = idx // nx, idx % nx
        pred[i] = pred_grid[row_idx * nx + col_idx]
    return pred


def run_refined_cmaq_guided(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    RefinedCMAQGuided: 精细化CMAQGuidedFusion
    """
    print("\n=== Refined CMAQ Guided Fusion ===")

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
        vna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, vna_pred_grid, ny, nx)
        avna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, avna_pred_grid, ny, nx)

        # 精细化权重
        obs_mean = train_df['Conc'].mean()
        cmaq_test = test_df['CMAQ'].values
        relative_dev = (cmaq_test - obs_mean) / obs_mean
        weight_vna = np.clip(1 / (1 + np.exp(-3 * relative_dev)), 0.25, 0.75)

        rcgf_pred = weight_vna * vna_pred + (1 - weight_vna) * avna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'rcgf': rcgf_pred
        }

        print(f"  Fold {fold_id}: completed")

    rcgf_all = np.concatenate([results[f]['rcgf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, rcgf_all)
    print(f"  Refined CMAQ Guided: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_triple_ensemble(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    TripleEnsemble: VNA + aVNA + eVNA 三重集成
    """
    print("\n=== Triple Ensemble Fusion ===")

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
        vna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, vna_pred_grid, ny, nx)
        avna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, avna_pred_grid, ny, nx)
        evna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, evna_pred_grid, ny, nx)

        # CMAQ引导的动态权重
        obs_mean = train_df['Conc'].mean()
        cmaq_test = test_df['CMAQ'].values
        relative_dev = (cmaq_test - obs_mean) / obs_mean

        w_vna = np.clip(0.5 + 0.5 * relative_dev, 0.2, 0.6)
        w_avna = np.clip(0.3 - 0.3 * relative_dev, 0.2, 0.5)
        w_evna = 1 - w_vna - w_avna
        w_evna = np.clip(w_evna, 0.1, 0.4)

        tef_pred = w_vna * vna_pred + w_avna * avna_pred + w_evna * evna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'tef': tef_pred
        }

        print(f"  Fold {fold_id}: completed")

    tef_all = np.concatenate([results[f]['tef'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, tef_all)
    print(f"  Triple Ensemble: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_cmaq_centroid_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    CMAQCentroidFusion: 基于CMAQ质心的融合
    """
    print("\n=== CMAQ Centroid Fusion ===")

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
        vna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, vna_pred_grid, ny, nx)
        avna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, avna_pred_grid, ny, nx)

        # 根据CMAQ分位数决定权重
        cmaq_quantiles = train_df['CMAQ'].quantile([0.25, 0.5, 0.75]).values
        cmaq_test = test_df['CMAQ'].values

        weight_vna = np.ones(len(test_df)) * 0.5
        low_mask = cmaq_test < cmaq_quantiles[0]
        weight_vna[low_mask] = 0.3
        high_mask = cmaq_test > cmaq_quantiles[2]
        weight_vna[high_mask] = 0.7
        mid_mask = (cmaq_test >= cmaq_quantiles[0]) & (cmaq_test <= cmaq_quantiles[2])
        weight_vna[mid_mask] = 0.5

        ccf_pred = weight_vna * vna_pred + (1 - weight_vna) * avna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'ccf': ccf_pred
        }

        print(f"  Fold {fold_id}: completed")

    ccf_all = np.concatenate([results[f]['ccf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, ccf_all)
    print(f"  CMAQ Centroid Fusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_distance_based_correction(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    DistanceBasedCorrection: 基于距离的偏差校正
    """
    print("\n=== Distance Based Correction Fusion ===")

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

        # 计算网格到最近监测站的距离
        nn_geo = NearestNeighbors(n_neighbors=1)
        nn_geo.fit(train_df[['Lon', 'Lat']].values)
        dist_to_station, _ = nn_geo.kneighbors(X_grid_full)

        max_dist = float(dist_to_station.max())
        normalized_dist = dist_to_station.ravel() / max_dist

        dist_weight = np.clip(0.5 + 0.3 * (1 - normalized_dist), 0.3, 0.7)

        fusion_grid = dist_weight * vna_pred_grid + (1 - dist_weight) * avna_pred_grid

        dbc_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, fusion_grid, ny, nx)

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'dbc': dbc_pred
        }

        print(f"  Fold {fold_id}: completed")

    dbc_all = np.concatenate([results[f]['dbc'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, dbc_all)
    print(f"  Distance Based Correction: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_combined_guided_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    CombinedGuidedFusion: 结合多种引导策略
    """
    print("\n=== Combined Guided Fusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    cmaq_gradient = np.sqrt(np.gradient(cmaq_year_avg, axis=0) ** 2 + np.gradient(cmaq_year_avg, axis=1) ** 2)
    cmaq_gradient_grid = cmaq_gradient.ravel()
    max_grad = float(cmaq_gradient_grid.max())

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

        # 网格到最近站的距离
        nn_geo = NearestNeighbors(n_neighbors=1)
        nn_geo.fit(train_df[['Lon', 'Lat']].values)
        dist_to_station, _ = nn_geo.kneighbors(X_grid_full)
        max_dist = float(dist_to_station.max())
        normalized_dist = dist_to_station.ravel() / max_dist

        # 多种权重
        obs_mean = train_df['Conc'].mean()
        obs_std = train_df['Conc'].std()

        # 计算CMAQ权重网格
        cmaq_dev_grid = y_grid_model_full - obs_mean
        relative_dev_grid = cmaq_dev_grid / obs_mean
        w1_vna_grid = np.clip(0.5 + 0.3 * relative_dev_grid, 0.25, 0.75)

        dist_weight_grid = np.clip(0.5 + 0.3 * (1 - normalized_dist), 0.3, 0.7)
        gradient_weight_grid = np.clip(0.5 + 0.2 * (1 - cmaq_gradient_grid / (max_grad + 1e-6)), 0.3, 0.7)

        final_weight = 0.5 * w1_vna_grid + 0.25 * dist_weight_grid + 0.25 * gradient_weight_grid
        final_weight = np.clip(final_weight, 0.25, 0.75)

        fusion_grid = final_weight * vna_pred_grid + (1 - final_weight) * avna_pred_grid

        cgf_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, fusion_grid, ny, nx)

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'cgf': cgf_pred
        }

        print(f"  Fold {fold_id}: completed")

    cgf_all = np.concatenate([results[f]['cgf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, cgf_all)
    print(f"  Combined Guided Fusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_final_ensemble(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    FinalEnsemble: 最终集成最佳方法
    """
    print("\n=== Final Ensemble Fusion ===")

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
        vna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, vna_pred_grid, ny, nx)
        avna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, avna_pred_grid, ny, nx)

        # CMAQ引导权重
        obs_mean = train_df['Conc'].mean()
        cmaq_test = test_df['CMAQ'].values
        deviation = np.abs(cmaq_test - obs_mean) / obs_mean
        weight_vna = np.clip(1 - deviation, 0.3, 0.7)

        cgf_pred = weight_vna * vna_pred + (1 - weight_vna) * avna_pred

        # Variance Weighted
        cmaq_gradient = np.gradient(cmaq_year_avg, axis=0) ** 2 + np.gradient(cmaq_year_avg, axis=1) ** 2
        cmaq_variance_grid = cmaq_gradient.ravel()
        variance_weight = cmaq_variance_grid / (cmaq_variance_grid.max() + 1e-6)
        vna_weight_vwf = np.clip(0.5 + 0.3 * variance_weight, 0.3, 0.7)
        vwf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            row_idx, col_idx = idx // nx, idx % nx
            w = vna_weight_vwf[row_idx * nx + col_idx]
            vwf_pred[i] = w * vna_pred[i] + (1 - w) * avna_pred[i]

        # MultiScale VNA
        scales = [
            {'k': 5, 'power': -2, 'weight': 0.2},
            {'k': 15, 'power': -2, 'weight': 0.4},
            {'k': 30, 'power': -2, 'weight': 0.4},
        ]
        ms_vna_preds = []
        for scale in scales:
            nn = NNA(method='nearest', k=scale['k'], power=scale['power'])
            nn.fit(train_df[['Lon', 'Lat']].values, train_df['Conc'].values)
            pred_grid = nn.predict(X_grid_full)
            ms_vna_preds.append(scale['weight'] * extract_grid_pred(test_df, lon_cmaq, lat_cmaq, pred_grid, ny, nx))
        ms_vna_pred = np.sum(ms_vna_preds, axis=0)

        weight_vna_ms = np.clip(1 - 0.5 * deviation, 0.3, 0.7)
        msf_pred = weight_vna_ms * ms_vna_pred + (1 - weight_vna_ms) * avna_pred

        # 最终集成
        fef_pred = 0.5 * cgf_pred + 0.3 * vwf_pred + 0.2 * msf_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'fef': fef_pred
        }

        print(f"  Fold {fold_id}: completed")

    fef_all = np.concatenate([results[f]['fef'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, fef_all)
    print(f"  Final Ensemble: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("年平均数据高级融合方法测试 (第四轮)")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    methods = [
        ('RefinedCMAQGuided', run_refined_cmaq_guided),
        ('TripleEnsemble', run_triple_ensemble),
        ('CMAQCentroidFusion', run_cmaq_centroid_fusion),
        ('DistanceBasedCorrection', run_distance_based_correction),
        ('CombinedGuidedFusion', run_combined_guided_fusion),
        ('FinalEnsemble', run_final_ensemble),
    ]

    for name, func in methods:
        metrics, _ = func(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
        results[name] = metrics

    print("\n" + "="*60)
    print("第四轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/fourth_round_results.csv', index=False)

    print("\n方法排名 (按R2降序):")
    for _, row in results_df.iterrows():
        print(f"  {row['method']:25s}: R2={row['R2']:.4f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}")

    return results_df


if __name__ == '__main__':
    results_df = main()