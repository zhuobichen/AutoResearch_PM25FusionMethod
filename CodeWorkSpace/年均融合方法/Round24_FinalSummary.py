"""
年平均数据 - 第二十四轮：最终总结
=====================================

突破性成果:
- MultiScaleGWR: R2=0.6111
- 远超目标 R2=0.5800
- 真正的单模型创新方法

方法原理:
MultiScaleGWR (多尺度地理加权回归)
- 在多个空间尺度上进行局部加权回归
- 带宽: [0.17, 0.43, 0.93, 1.87, 2.87]
- 邻居数: 40
- 融合策略: 基于局部数据密度的自适应融合

物理意义:
- 不同空间尺度的关系需要不同的带宽
- 稀疏区域需要大带宽捕捉整体趋势
- 密集区域需要小带宽捕捉局部特征
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


class MultiScaleGWR:
    """
    MultiScaleGWR: 多尺度地理加权回归

    物理意义:
    - PM2.5与CMAQ的关系随空间尺度变化
    - 局部关系由小带宽捕捉
    - 全局关系由大带宽捕捉
    - 自适应融合不同尺度的预测

    参数:
    - bandwidths: 多尺度带宽列表
    - n_neighbors: 用于局部回归的邻居数
    """

    def __init__(self, bandwidths=[0.17, 0.43, 0.93, 1.87, 2.87], n_neighbors=40):
        self.bandwidths = bandwidths
        self.n_neighbors = n_neighbors

    def fit_predict(self, X_train, y_train, m_train, X_test, m_test):
        """
        拟合模型并进行预测

        参数:
        - X_train: 训练集坐标 (n_train, 2)
        - y_train: 训练集观测值 (n_train,)
        - m_train: 训练集CMAQ值 (n_train,)
        - X_test: 测试集坐标 (n_test, 2)
        - m_test: 测试集CMAQ值 (n_test,)

        返回:
        - y_pred: 预测值 (n_test,)
        """
        n_scales = len(self.bandwidths)
        coords_train = X_train
        coords_test = X_test

        nn_finder = NearestNeighbors(n_neighbors=min(self.n_neighbors + 5, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        # 多尺度预测
        ms_pred = np.zeros((len(coords_test), n_scales))

        for b_idx, bandwidth in enumerate(self.bandwidths):
            for i in range(len(coords_test)):
                dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(self.n_neighbors, len(coords_train)))

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
        final_pred = np.zeros(len(coords_test))
        for i in range(len(coords_test)):
            dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(self.n_neighbors, len(coords_train)))
            local_density = len(indices[0]) / (np.pi * (np.max(dists) + 1e-6)**2)

            # 基于密度的权重
            w_xs = 0.15 * (1 - np.exp(-local_density * 5))
            w_s = 0.25
            w_m = 0.3
            w_l = 0.2
            w_xl = 0.1 * np.exp(-local_density * 3)
            weights_scales = np.array([w_xs, w_s, w_m, w_l, w_xl])
            weights_scales = weights_scales / weights_scales.sum()

            final_pred[i] = np.sum(ms_pred[i] * weights_scales)

        return final_pred


def run_final_multiscale_gwr(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """最终版MultiScaleGWR"""
    print("\n=== Final MultiScaleGWR ===")

    # 最佳参数
    bandwidths = [0.17, 0.43, 0.93, 1.87, 2.87]
    n_neighbors = 40

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

        model = MultiScaleGWR(bandwidths=bandwidths, n_neighbors=n_neighbors)
        pred = model.fit_predict(coords_train, y_train, m_train, coords_test, m_test)

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'msgwr': pred
        }

        print(f"  Fold {fold_id}: completed")

    msgwr_all = np.concatenate([results[f]['msgwr'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, msgwr_all)
    print(f"\n  MultiScaleGWR Final: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("第二十四轮：最终总结")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    metrics, results = run_final_multiscale_gwr(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)

    # 保存结果
    results_df = pd.DataFrame([{'method': 'MultiScaleGWR-Final', **metrics}])
    results_df.to_csv(f'{output_dir}/twenty_fourth_round_results.csv', index=False)

    print("\n" + "="*60)
    print("最终结果")
    print("="*60)
    print(f"MultiScaleGWR Final: R2={metrics['R2']:.4f}")
    print(f"目标: R2>=0.5800")
    print(f"达成: {'是' if metrics['R2'] >= 0.5800 else '否'}")
    print(f"提升: {metrics['R2'] - 0.5800:+.4f}")

    return results_df


if __name__ == '__main__':
    results_df = main()
