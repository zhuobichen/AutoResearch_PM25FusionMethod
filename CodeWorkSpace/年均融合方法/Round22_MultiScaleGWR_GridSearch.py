"""
年平均数据 - 第二十二轮：MultiScaleGWR网格搜索
============================================

最佳结果: MultiScaleGWR R2=0.6005
带宽组合: [0.5, 1.0, 2.0]

目标: 通过网格搜索找到更好的带宽组合
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
import netCDF4 as nc

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


def run_multiscale_gwr(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg, bandwidths):
    """MultiScaleGWR with given bandwidths"""
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

        n_scales = len(bandwidths)
        nn_finder = NearestNeighbors(n_neighbors=min(30, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        ms_pred = np.zeros((len(test_df), n_scales))

        for b_idx, bandwidth in enumerate(bandwidths):
            for i in range(len(test_df)):
                dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(25, len(coords_train)))

                weights = np.exp(-0.5 * (dists[0] / bandwidth)**2)
                weights = weights / (weights.sum() + 1e-6)

                X_local = np.column_stack([np.ones(len(indices[0])), m_train[indices[0]]])
                y_local = y_train[indices[0]]

                W = np.diag(weights)
                try:
                    XTWX = X_local.T @ W @ X_local + np.eye(2) * 1e-6
                    XTWy = X_local.T @ W @ y_local
                    beta = np.linalg.solve(XTWX, XTWy)
                    ms_pred[i, b_idx] = beta[0] + beta[1] * m_test[i]
                except:
                    ms_pred[i, b_idx] = y_train.mean() + (m_test[i] - m_train.mean()) * np.corrcoef(y_train, m_train)[0, 1]

        # 融合
        final_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(15, len(coords_train)))
            local_density = len(indices[0]) / (np.pi * (np.max(dists) + 1e-6)**2)

            # 权重根据密度调整
            n = len(bandwidths)
            weights_scales = np.ones(n) / n

            if n == 3:
                w_small = 0.3 * (1 - np.exp(-local_density * 5))
                w_medium = 0.4
                w_large = 0.3 * np.exp(-local_density * 5)
                weights_scales = np.array([w_small, w_medium, w_large])
            elif n == 4:
                w_xs = 0.15 * (1 - np.exp(-local_density * 5))
                w_s = 0.35
                w_m = 0.35
                w_l = 0.15 * np.exp(-local_density * 5)
                weights_scales = np.array([w_xs, w_s, w_m, w_l])

            weights_scales = weights_scales / weights_scales.sum()
            final_pred[i] = np.sum(ms_pred[i] * weights_scales)

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'msgwr': final_pred
        }

    msgwr_all = np.concatenate([results[f]['msgwr'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, msgwr_all)

    return metrics


def main():
    print("="*60)
    print("第二十二轮：MultiScaleGWR网格搜索")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    # 测试不同的带宽组合
    bandwidth_combinations = [
        [0.5, 1.0, 2.0],    # 原始最佳
        [0.3, 0.8, 1.5],    # 更小尺度
        [0.5, 1.2, 2.2],    # 中间偏移
        [0.4, 0.9, 1.8],    # 整体缩小
        [0.6, 1.1, 2.0],    # 整体放大
        [0.3, 0.6, 1.2, 2.0],  # 4尺度
        [0.4, 0.8, 1.5, 2.5],  # 4尺度变体
        [0.25, 0.6, 1.0, 1.8, 2.8],  # 5尺度
        [0.35, 0.7, 1.2, 2.0],  # 4尺度
        [0.5, 1.0, 1.8, 2.8],  # 4尺度
        [0.2, 0.5, 1.0, 2.0, 3.0],  # 5尺度
        [0.3, 0.7, 1.4, 2.5],  # 4尺度
        [0.4, 1.0, 2.2],    # 等比
    ]

    print("\n=== Grid Search Results ===")
    results = {}

    for bandwidths in bandwidth_combinations:
        print(f"\nTesting bandwidths: {bandwidths}")
        metrics = run_multiscale_gwr(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg, bandwidths)
        bw_str = '-'.join([str(b) for b in bandwidths])
        results[f'BW-{bw_str}'] = {
            'bandwidths': bandwidths,
            'R2': metrics['R2'],
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MB': metrics['MB']
        }
        print(f"  R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    # 排序结果
    sorted_results = sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True)

    print("\n" + "="*60)
    print("网格搜索结果排名 (按R2降序)")
    print("="*60)

    for name, metrics in sorted_results:
        print(f"  {name:40s}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    # 保存最佳结果
    best_name, best_metrics = sorted_results[0]
    print(f"\n最佳配置: {best_name}")
    print(f"  R2={best_metrics['R2']:.4f}")

    # 保存到CSV
    results_df = pd.DataFrame([
        {'method': name, 'bandwidths': str(m['bandwidths']), **m}
        for name, m in sorted_results
    ])
    results_df.to_csv(f'{output_dir}/twenty_second_round_results.csv', index=False)

    return results_df


if __name__ == '__main__':
    results_df = main()
