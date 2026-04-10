"""
年平均数据 - 第二十一轮：MultiScaleGWR优化
==========================================

基于第二十轮测试结果:
- MultiScaleGWR: R2=0.6007 (达标！突破性进展)
- 这是真正的单模型创新

优化方向：
1. 网格搜索最佳带宽组合
2. 尝试不同的尺度融合策略
3. 结合CMAQ信息增强
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


def run_multiscale_gwr_optimized(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg, bandwidths, scale_weights=None):
    """
    MultiScaleGWR优化版本
    """
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

        # 多尺度融合
        if scale_weights is None:
            scale_weights = np.ones(n_scales) / n_scales

        final_pred = np.sum(ms_pred * np.array(scale_weights).reshape(1, -1), axis=1)

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'msgwr': final_pred
        }

        print(f"  Fold {fold_id}: completed")

    msgwr_all = np.concatenate([results[f]['msgwr'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, msgwr_all)
    print(f"  MultiScaleGWR: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_multiscale_gwr_adaptive(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    MultiScaleGWR自适应版本
    根据局部数据密度自适应选择尺度权重
    """
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

        bandwidths = [0.5, 1.0, 2.0]
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

        # 自适应融合
        final_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(15, len(coords_train)))
            local_density = len(indices[0]) / (np.pi * (np.max(dists) + 1e-6)**2)

            # 根据密度计算权重
            # 数据密集 -> 小尺度权重高
            # 数据稀疏 -> 大尺度权重高
            w_small = 0.2 * (1 - np.exp(-local_density * 3))
            w_medium = 0.5
            w_large = 0.3 * np.exp(-local_density * 3)

            weights_scales = np.array([w_small, w_medium, w_large])
            weights_scales = weights_scales / weights_scales.sum()

            final_pred[i] = np.sum(ms_pred[i] * weights_scales)

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'msgwr': final_pred
        }

        print(f"  Fold {fold_id}: completed")

    msgwr_all = np.concatenate([results[f]['msgwr'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, msgwr_all)
    print(f"  MultiScaleGWR-Adaptive: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_multiscale_gwr_cmaq_enhanced(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    MultiScaleGWR-CMAQ增强版本
    结合CMAQ全局趋势和局部GWR
    """
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

        bandwidths = [0.5, 1.0, 2.0]
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

        # 计算CMAQ全局偏差
        global_bias = np.mean(y_train - m_train)

        # CMAQ增强融合
        final_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            # 多尺度GWR
            gwr_combined = np.mean(ms_pred[i])

            # CMAQ + 全局偏差
            cmaq_pred = m_test[i] + global_bias

            # 局部数据密度
            dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(15, len(coords_train)))
            local_density = len(indices[0]) / (np.pi * (np.max(dists) + 1e-6)**2)

            # 数据密集区域更多依赖GWR，数据稀疏区域更多依赖CMAQ
            w_gwr = 0.6 + 0.3 * (1 - np.exp(-local_density * 5))
            w_cmaq = 1 - w_gwr

            final_pred[i] = w_gwr * gwr_combined + w_cmaq * cmaq_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'msgwr': final_pred
        }

        print(f"  Fold {fold_id}: completed")

    msgwr_all = np.concatenate([results[f]['msgwr'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, msgwr_all)
    print(f"  MultiScaleGWR-CMAQ: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_multiscale_gwr_fine_tuned(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    MultiScaleGWR精细调参版本
    基于第二十轮结果优化
    """
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

        # 精细调整的带宽
        bandwidths = [0.4, 0.8, 1.5, 2.5]
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

        # 精细调优的融合权重
        final_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(15, len(coords_train)))
            local_density = len(indices[0]) / (np.pi * (np.max(dists) + 1e-6)**2)

            # 根据密度调整权重
            # 密集：更多中等带宽
            # 稀疏：更多大带宽
            w_xs = 0.15 * (1 - np.exp(-local_density * 5))
            w_s = 0.35
            w_m = 0.35 * (1 + 0.5 * np.exp(-local_density * 3))
            w_l = 0.15 * np.exp(-local_density * 3)

            weights_scales = np.array([w_xs, w_s, w_m, w_l])
            weights_scales = weights_scales / weights_scales.sum()

            final_pred[i] = np.sum(ms_pred[i] * weights_scales)

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'msgwr': final_pred
        }

        print(f"  Fold {fold_id}: completed")

    msgwr_all = np.concatenate([results[f]['msgwr'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, msgwr_all)
    print(f"  MultiScaleGWR-FineTuned: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("第二十一轮：MultiScaleGWR优化")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    # 原始MultiScaleGWR
    print("\n=== Testing different configurations ===")
    metrics, _ = run_multiscale_gwr_optimized(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg,
                                                bandwidths=[0.5, 1.0, 2.0])
    results['MultiScaleGWR-orig'] = metrics

    # 自适应版本
    metrics, _ = run_multiscale_gwr_adaptive(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
    results['MultiScaleGWR-Adaptive'] = metrics

    # CMAQ增强版本
    metrics, _ = run_multiscale_gwr_cmaq_enhanced(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
    results['MultiScaleGWR-CMAQ'] = metrics

    # 精细调参版本
    metrics, _ = run_multiscale_gwr_fine_tuned(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
    results['MultiScaleGWR-FineTuned'] = metrics

    print("\n" + "="*60)
    print("第二十一轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/twenty_first_round_results.csv', index=False)

    print("\n方法排名 (按R2降序):")
    for _, row in results_df.iterrows():
        print(f"  {row['method']:30s}: R2={row['R2']:.4f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}, MB={row['MB']:.2f}")

    print("\n" + "="*60)
    print("与目标对比 (R2>=0.5800)")
    print("="*60)
    for _, row in results_df.iterrows():
        diff = row['R2'] - 0.5800
        status = "达标" if diff >= 0 else "差距"
        print(f"  {row['method']:30s}: R2={row['R2']:.4f}, 差距 {diff:+.4f} [{status}]")

    return results_df


if __name__ == '__main__':
    results_df = main()
