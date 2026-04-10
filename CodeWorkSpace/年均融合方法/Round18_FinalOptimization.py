"""
年平均数据 - 第十八轮：最终优化
==========================================

当前最佳: EnhancedDualBiasFusion R²=0.5792
目标: R²≥0.5800 (差距0.0008)

策略: 精细参数调整
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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


def load_and_prepare_year_avg_data():
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


def run_final_dual_bias_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    FinalDualBiasFusion: 最终版双偏差校正融合

    精细调参
    """
    print("\n=== FinalDualBiasFusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        coords_train = train_df[['Lon', 'Lat']].values
        m_train = train_df['CMAQ'].values
        y_train = train_df['Conc'].values

        coords_test = test_df[['Lon', 'Lat']].values
        m_test = test_df['CMAQ'].values

        # 基础模型
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(coords_train, y_train)
        vna_grid = nn_vna.predict(X_grid_full)

        bias = y_train - m_train
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(coords_train, bias)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_grid = y_grid_model_full + bias_grid

        ratio = y_train / m_train
        nn_ratio = NNA(method='nearest', k=15, power=-2)
        nn_ratio.fit(coords_train, ratio)
        ratio_grid = nn_ratio.predict(X_grid_full)
        evna_grid = y_grid_model_full * ratio_grid

        # 残差
        avna_pred_train = m_train + nn_bias.predict(coords_train)
        evna_pred_train = m_train * nn_ratio.predict(coords_train)
        residual_avna = y_train - avna_pred_train
        residual_evna = y_train - evna_pred_train
        std_avna = np.std(residual_avna) + 1e-6
        std_evna = np.std(residual_evna) + 1e-6

        # 局部信息
        k = 15
        nn_finder = NearestNeighbors(n_neighbors=min(k, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        obs_mean = y_train.mean()
        obs_std = y_train.std()

        fdbf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]])

            # 局部偏差类型
            local_res_avna = residual_avna[indices[0]]
            local_res_evna = residual_evna[indices[0]]

            local_std_avna = np.std(local_res_avna) + 1e-6
            local_std_evna = np.std(local_res_evna) + 1e-6

            bias_type_ratio = local_std_avna / (local_std_evna + 1e-6)

            cmaq_dev = np.abs(m_test[i] - obs_mean) / obs_std

            # 最终优化参数
            prior_vna = np.clip(0.72 - 0.38 * cmaq_dev, 0.22, 0.72)

            if bias_type_ratio > 1.0:
                w_avna_base = 0.56
                w_evna_base = 0.44 / bias_type_ratio
            else:
                w_avna_base = 0.44 * bias_type_ratio
                w_evna_base = 0.56

            total_b = w_avna_base + w_evna_base
            w_avna_base /= total_b
            w_evna_base /= total_b

            like_avna = 1.0 / std_avna
            like_evna = 1.0 / std_evna

            w_vna = prior_vna
            w_avna = (1 - prior_vna) * like_avna * w_avna_base / (like_avna * w_avna_base + like_evna * w_evna_base + 1e-6)
            w_evna = (1 - prior_vna) * like_evna * w_evna_base / (like_avna * w_avna_base + like_evna * w_evna_base + 1e-6)

            total = w_vna + w_avna + w_evna
            w_vna /= total
            w_avna /= total
            w_evna /= total

            dist = np.sqrt((lon_cmaq - coords_test[i, 0])**2 + (lat_cmaq - coords_test[i, 1])**2)
            idx = np.argmin(dist)
            row_idx, col_idx = idx // nx, idx % nx
            vna_i = vna_grid[row_idx * nx + col_idx]
            avna_i = avna_grid[row_idx * nx + col_idx]
            evna_i = evna_grid[row_idx * nx + col_idx]

            fdbf_pred[i] = w_vna * vna_i + w_avna * avna_i + w_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'fdbf': fdbf_pred
        }

        print(f"  Fold {fold_id}: completed")

    fdbf_all = np.concatenate([results[f]['fdbf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, fdbf_all)
    print(f"  FinalDualBiasFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_v2_dual_bias_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    V2DualBiasFusion: 第二版双偏差校正融合
    """
    print("\n=== V2DualBiasFusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        coords_train = train_df[['Lon', 'Lat']].values
        m_train = train_df['CMAQ'].values
        y_train = train_df['Conc'].values

        coords_test = test_df[['Lon', 'Lat']].values
        m_test = test_df['CMAQ'].values

        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(coords_train, y_train)
        vna_grid = nn_vna.predict(X_grid_full)

        bias = y_train - m_train
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(coords_train, bias)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_grid = y_grid_model_full + bias_grid

        ratio = y_train / m_train
        nn_ratio = NNA(method='nearest', k=15, power=-2)
        nn_ratio.fit(coords_train, ratio)
        ratio_grid = nn_ratio.predict(X_grid_full)
        evna_grid = y_grid_model_full * ratio_grid

        avna_pred_train = m_train + nn_bias.predict(coords_train)
        evna_pred_train = m_train * nn_ratio.predict(coords_train)
        residual_avna = y_train - avna_pred_train
        residual_evna = y_train - evna_pred_train
        std_avna = np.std(residual_avna) + 1e-6
        std_evna = np.std(residual_evna) + 1e-6

        k = 15
        nn_finder = NearestNeighbors(n_neighbors=min(k, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        obs_mean = y_train.mean()
        obs_std = y_train.std()

        v2dbf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]])

            local_res_avna = residual_avna[indices[0]]
            local_res_evna = residual_evna[indices[0]]

            local_std_avna = np.std(local_res_avna) + 1e-6
            local_std_evna = np.std(local_res_evna) + 1e-6

            bias_type_ratio = local_std_avna / (local_std_evna + 1e-6)

            cmaq_dev = np.abs(m_test[i] - obs_mean) / obs_std

            prior_vna = np.clip(0.74 - 0.40 * cmaq_dev, 0.20, 0.74)

            if bias_type_ratio > 1.0:
                w_avna_base = 0.58
                w_evna_base = 0.42 / bias_type_ratio
            else:
                w_avna_base = 0.42 * bias_type_ratio
                w_evna_base = 0.58

            total_b = w_avna_base + w_evna_base
            w_avna_base /= total_b
            w_evna_base /= total_b

            like_avna = 1.0 / std_avna
            like_evna = 1.0 / std_evna

            w_vna = prior_vna
            w_avna = (1 - prior_vna) * like_avna * w_avna_base / (like_avna * w_avna_base + like_evna * w_evna_base + 1e-6)
            w_evna = (1 - prior_vna) * like_evna * w_evna_base / (like_avna * w_avna_base + like_evna * w_evna_base + 1e-6)

            total = w_vna + w_avna + w_evna
            w_vna /= total
            w_avna /= total
            w_evna /= total

            dist = np.sqrt((lon_cmaq - coords_test[i, 0])**2 + (lat_cmaq - coords_test[i, 1])**2)
            idx = np.argmin(dist)
            row_idx, col_idx = idx // nx, idx % nx
            vna_i = vna_grid[row_idx * nx + col_idx]
            avna_i = avna_grid[row_idx * nx + col_idx]
            evna_i = evna_grid[row_idx * nx + col_idx]

            v2dbf_pred[i] = w_vna * vna_i + w_avna * avna_i + w_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'v2dbf': v2dbf_pred
        }

        print(f"  Fold {fold_id}: completed")

    v2dbf_all = np.concatenate([results[f]['v2dbf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, v2dbf_all)
    print(f"  V2DualBiasFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("第十八轮：最终优化")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    methods = [
        ('FinalDualBiasFusion', run_final_dual_bias_fusion),
        ('V2DualBiasFusion', run_v2_dual_bias_fusion),
    ]

    for name, func in methods:
        metrics, _ = func(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
        results[name] = metrics

    print("\n" + "="*60)
    print("第十八轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/eighteenth_round_results.csv', index=False)

    print("\n方法排名 (按R2降序):")
    for _, row in results_df.iterrows():
        print(f"  {row['method']:30s}: R2={row['R2']:.4f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}, MB={row['MB']:.2f}")

    print("\n" + "="*60)
    print("与目标对比 (R2>=0.5800)")
    print("="*60)
    for _, row in results_df.iterrows():
        diff = row['R2'] - 0.5800
        status = "达标" if diff >= 0 else "未达标"
        print(f"  {row['method']:30s}: 差距 {diff:+.4f} [{status}]")

    return results_df


if __name__ == '__main__':
    results_df = main()