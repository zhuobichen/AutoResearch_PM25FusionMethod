"""
年平均数据 - 第二十三轮：MultiScaleGWR最终优化
============================================

最佳结果: Bandwidths=[0.2, 0.5, 1.0, 2.0, 3.0] R2=0.6063

目标: 进一步优化
1. 围绕最佳带宽微调
2. 尝试不同的邻居数
3. 尝试不同的融合策略
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


def run_multiscale_gwr(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg, bandwidths, n_neighbors=25):
    """MultiScaleGWR with given bandwidths and n_neighbors"""
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
        nn_finder = NearestNeighbors(n_neighbors=min(n_neighbors + 5, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        ms_pred = np.zeros((len(test_df), n_scales))

        for b_idx, bandwidth in enumerate(bandwidths):
            for i in range(len(test_df)):
                dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(n_neighbors, len(coords_train)))

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
            dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(n_neighbors, len(coords_train)))
            local_density = len(indices[0]) / (np.pi * (np.max(dists) + 1e-6)**2)

            # 权重根据密度调整
            n = len(bandwidths)
            weights_scales = np.ones(n) / n

            if n == 5:
                # 针对5个尺度的权重
                w_xs = 0.15 * (1 - np.exp(-local_density * 5))
                w_s = 0.25
                w_m = 0.3
                w_l = 0.2
                w_xl = 0.1 * np.exp(-local_density * 3)
                weights_scales = np.array([w_xs, w_s, w_m, w_l, w_xl])

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


def run_multiscale_gwr_v2(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg, bandwidths, n_neighbors=25):
    """MultiScaleGWR v2 - 不同的融合策略"""
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
        nn_finder = NearestNeighbors(n_neighbors=min(n_neighbors + 5, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        ms_pred = np.zeros((len(test_df), n_scales))

        for b_idx, bandwidth in enumerate(bandwidths):
            for i in range(len(test_df)):
                dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(n_neighbors, len(coords_train)))

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

        # 融合v2: 基于局部数据质量的加权
        final_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            # 计算每个尺度的局部预测质量估计
            dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(n_neighbors, len(coords_train)))

            # 使用均方根距离作为质量指标
            rms_dist = np.sqrt(np.mean(dists**2))
            local_density = len(indices[0]) / (np.pi * (np.max(dists) + 1e-6)**2)

            # 带宽越小，局部性越强，但受稀疏性影响越大
            # 融合时，给更稳定（中等带宽）的尺度更高权重
            n = len(bandwidths)
            weights_scales = np.ones(n) / n

            if n == 5:
                # 中等带宽权重最高
                weights_scales = np.array([0.15, 0.25, 0.35, 0.15, 0.10])

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


def run_multiscale_gwr_v3(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg, bandwidths, n_neighbors=25):
    """MultiScaleGWR v3 - 结合CMAQ偏差"""
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
        nn_finder = NearestNeighbors(n_neighbors=min(n_neighbors + 5, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        ms_pred = np.zeros((len(test_df), n_scales))

        for b_idx, bandwidth in enumerate(bandwidths):
            for i in range(len(test_df)):
                dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(n_neighbors, len(coords_train)))

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

        # 融合v3: GWR + CMAQ全局校正
        final_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            # 多尺度GWR预测
            gwr_pred = np.mean(ms_pred[i])

            # CMAQ + 全局偏差
            global_bias = np.mean(y_train - m_train)
            cmaq_pred = m_test[i] + global_bias

            # 根据CMAQ偏差调整
            cmaq_dev = np.abs(m_test[i] - m_train.mean()) / m_train.std()

            # CMAQ偏差大时，更多依赖GWR
            w_gwr = 0.7 + 0.2 * cmaq_dev
            w_cmaq = 1 - w_gwr

            final_pred[i] = w_gwr * gwr_pred + w_cmaq * cmaq_pred

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
    print("第二十三轮：MultiScaleGWR最终优化")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    print("\n=== Testing different configurations ===")

    results = {}

    # 最佳带宽 + 不同邻居数
    best_bw = [0.2, 0.5, 1.0, 2.0, 3.0]
    for n_neighbors in [15, 20, 25, 30]:
        print(f"\nTesting BW={best_bw}, n_neighbors={n_neighbors}")
        metrics = run_multiscale_gwr(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg, best_bw, n_neighbors)
        results[f'BW={best_bw}, k={n_neighbors}'] = metrics
        print(f"  R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    # 不同带宽 + 最佳邻居数
    bandwidths_to_test = [
        [0.15, 0.4, 0.9, 1.8, 2.8],  # 缩小
        [0.25, 0.55, 1.1, 2.1, 3.1],  # 微调
        [0.1, 0.4, 0.9, 1.8, 2.8],  # 更小起始
        [0.2, 0.5, 1.1, 2.2, 3.5],  # 放大结束
    ]

    for bandwidths in bandwidths_to_test:
        print(f"\nTesting BW={bandwidths}, k=25")
        metrics = run_multiscale_gwr(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg, bandwidths, n_neighbors=25)
        results[f'BW={bandwidths}'] = metrics
        print(f"  R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    # v2版本测试
    print(f"\nTesting v2 (固定权重) BW={best_bw}, k=25")
    metrics_v2 = run_multiscale_gwr_v2(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg, best_bw, n_neighbors=25)
    results['v2-fixed'] = metrics_v2
    print(f"  R2={metrics_v2['R2']:.4f}, MAE={metrics_v2['MAE']:.2f}, RMSE={metrics_v2['RMSE']:.2f}")

    # v3版本测试
    print(f"\nTesting v3 (CMAQ偏差) BW={best_bw}, k=25")
    metrics_v3 = run_multiscale_gwr_v3(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg, best_bw, n_neighbors=25)
    results['v3-cmaq'] = metrics_v3
    print(f"  R2={metrics_v3['R2']:.4f}, MAE={metrics_v3['MAE']:.2f}, RMSE={metrics_v3['RMSE']:.2f}")

    # 排序结果
    sorted_results = sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True)

    print("\n" + "="*60)
    print("优化结果排名 (按R2降序)")
    print("="*60)

    for name, metrics in sorted_results:
        print(f"  {name:50s}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    # 保存结果
    results_df = pd.DataFrame([
        {'method': name, **m} for name, m in sorted_results
    ])
    results_df.to_csv(f'{output_dir}/twenty_third_round_results.csv', index=False)

    return results_df


if __name__ == '__main__':
    results_df = main()
