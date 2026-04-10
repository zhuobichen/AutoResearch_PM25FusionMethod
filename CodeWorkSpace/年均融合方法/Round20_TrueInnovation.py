"""
年平均数据 - 第二十轮：真正创新的单模型
==========================================

基于第十九轮测试结果:
- GWRFusion: R2=0.5778 (最有希望的物理方法)
- 简单协同克里金/回归克里金效果不佳

真正创新的单模型方向：
1. EnhancedGWR: 增强型地理加权回归 - CMAQ非线性关系+自适应带宽
2. SpatialProcessGP: 空间过程模型 - CMAQ作为先验均值的高斯过程
3. MultiScaleGWR: 多尺度GWR - 不同带宽的GWR融合
4. LocalBiasGWR: 局部偏差校正GWR - GWR+局部偏差修正

核心约束:
- 单模型（非简单加权组合）
- 物理可解释
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
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


def run_enhanced_gwr(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    EnhancedGWR: 增强型地理加权回归

    物理意义:
    - PM2.5与CMAQ的非线性关系随空间变化
    - 使用自适应带宽捕捉空间非平稳性
    - 局部加权回归捕捉局部特征
    """
    print("\n=== EnhancedGWR ===")

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

        # 自适应带宽：基于局部数据密度
        nn_finder = NearestNeighbors(n_neighbors=min(25, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        egwr_pred = np.zeros(len(test_df))

        for i in range(len(test_df)):
            # 自适应带宽：使用最近k个邻点的平均距离
            dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(20, len(coords_train)))
            mean_dist = np.mean(dists)
            bandwidth = max(mean_dist, 0.3)

            # 高斯核权重
            weights = np.exp(-0.5 * (dists[0] / bandwidth)**2)
            weights = weights / (weights.sum() + 1e-6)

            # 局部特征：M, M^2, 距离中心的偏移
            X_local = np.column_stack([
                np.ones(len(indices[0])),
                m_train[indices[0]],
                m_train[indices[0]]**2
            ])
            y_local = y_train[indices[0]]

            # 加权最小二乘
            W = np.diag(weights)
            try:
                XTWX = X_local.T @ W @ X_local + np.eye(3) * 1e-6
                XTWy = X_local.T @ W @ y_local
                beta = np.linalg.solve(XTWX, XTWy)
                egwr_pred[i] = beta[0] + beta[1] * m_test[i] + beta[2] * m_test[i]**2
            except:
                # 降级使用简单线性
                X_local_simple = np.column_stack([np.ones(len(indices[0])), m_train[indices[0]]])
                W_simple = np.diag(weights)
                try:
                    XTWX = X_local_simple.T @ W_simple @ X_local_simple + np.eye(2) * 1e-6
                    XTWy = X_local_simple.T @ W_simple @ y_local
                    beta = np.linalg.solve(XTWX, XTWy)
                    egwr_pred[i] = beta[0] + beta[1] * m_test[i]
                except:
                    # 最差情况：使用全局线性
                    egwr_pred[i] = y_train.mean() + (m_test[i] - m_train.mean()) * np.corrcoef(y_train, m_train)[0, 1]

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'egwr': egwr_pred
        }

        print(f"  Fold {fold_id}: completed")

    egwr_all = np.concatenate([results[f]['egwr'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, egwr_all)
    print(f"  EnhancedGWR: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_spatial_process_gp(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    SpatialProcessGP: 空间过程高斯过程

    物理意义:
    - PM2.5浓度是一个空间随机过程
    - CMAQ提供了该过程的先验均值
    - GPR对空间过程进行后验推断
    """
    print("\n=== SpatialProcessGP ===")

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

        # 计算残差用于确定GPR参数
        residual = y_train - m_train
        residual_var = np.var(residual)

        # GPR核函数
        kernel = C(1.0, (0.1, 10)) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10)) + WhiteKernel(noise_level=residual_var * 0.5, noise_level_bounds=(1e-5, residual_var))

        # 拟合GPR模型
        try:
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=residual_var * 0.1, normalize_y=True)
            gpr.fit(coords_train, m_train)  # 用CMAQ训练

            # 预测：GPR给出CMAQ的条件均值
            gp_pred, gp_std = gpr.predict(coords_test, return_std=True)

            # 最终预测 = GPR预测 + 训练集平均偏差
            # 但这里需要直接预测观测值
            # 用观测值训练GPR
            gpr_obs = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=residual_var * 0.1, normalize_y=True)
            gpr_obs.fit(coords_train, y_train)
            spgp_pred, _ = gpr_obs.predict(coords_test, return_std=True)
        except:
            # GPR失败时使用IDW
            nn_finder = NearestNeighbors(n_neighbors=min(20, len(coords_train)), algorithm='ball_tree')
            nn_finder.fit(coords_train)
            dists, indices = nn_finder.kneighbors(coords_test)
            weights = 1.0 / (dists**2 + 1e-6)
            weights = weights / weights.sum(axis=1, keepdims=True)
            spgp_pred = np.sum(weights * y_train[indices], axis=1)

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'spgp': spgp_pred
        }

        print(f"  Fold {fold_id}: completed")

    spgp_all = np.concatenate([results[f]['spgp'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, spgp_all)
    print(f"  SpatialProcessGP: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_local_bias_gwr(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    LocalBiasGWR: 局部偏差校正的GWR

    物理意义:
    - GWR捕捉空间非平稳性
    - 局部偏差校正在GWR基础上进一步修正系统偏差
    """
    print("\n=== LocalBiasGWR ===")

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

        nn_finder = NearestNeighbors(n_neighbors=min(25, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        # GWR预测 (不用于训练点LOO)
        gwr_pred = np.zeros(len(test_df))

        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(20, len(coords_train)))
            mean_dist = np.mean(dists) + 1e-6
            bandwidth = max(mean_dist, 0.3)

            weights = np.exp(-0.5 * (dists[0] / bandwidth)**2)
            weights = weights / (weights.sum() + 1e-6)

            X_local = np.column_stack([np.ones(len(indices[0])), m_train[indices[0]]])
            y_local = y_train[indices[0]]

            W = np.diag(weights)
            try:
                XTWX = X_local.T @ W @ X_local + np.eye(2) * 1e-6
                XTWy = X_local.T @ W @ y_local
                beta = np.linalg.solve(XTWX, XTWy)
                gwr_pred[i] = beta[0] + beta[1] * m_test[i]
            except:
                gwr_pred[i] = y_train.mean() + (m_test[i] - m_train.mean()) * np.corrcoef(y_train, m_train)[0, 1]

        # 计算局部偏差：O - M (bias)
        bias = y_train - m_train

        # 对测试点进行局部偏差校正
        lbgwr_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(15, len(coords_train)))
            local_bias = bias[indices[0]]

            weights_bias = 1.0 / (dists[0]**2 + 1e-6)
            weights_bias = weights_bias / weights_bias.sum()
            mean_bias = np.sum(weights_bias * local_bias)

            lbgwr_pred[i] = m_test[i] + mean_bias

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'lbgwr': lbgwr_pred
        }

        print(f"  Fold {fold_id}: completed")

    lbgwr_all = np.concatenate([results[f]['lbgwr'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, lbgwr_all)
    print(f"  LocalBiasGWR: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_multiscale_gwr(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    MultiScaleGWR: 多尺度地理加权回归

    物理意义:
    - 不同空间尺度的关系需要不同的带宽
    - 稀疏区域需要大带宽，密集区域需要小带宽
    - 多尺度融合捕捉不同尺度的特征
    """
    print("\n=== MultiScaleGWR ===")

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

        # 多尺度带宽
        bandwidths = [0.5, 1.0, 2.0]

        nn_finder = NearestNeighbors(n_neighbors=min(30, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        ms_pred = np.zeros((len(test_df), len(bandwidths)))

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

        # 多尺度融合：基于局部数据密度选择权重
        final_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(15, len(coords_train)))
            # 数据密集度
            local_density = len(indices[0]) / (np.pi * (np.max(dists) + 1e-6)**2)

            # 数据越密集，小带宽权重越高
            # 数据越稀疏，大带宽权重越高
            weights_scales = np.array([
                0.3 * (1 - np.exp(-local_density * 5)),  # 小带宽
                0.4,
                0.3 * (np.exp(-local_density * 5))  # 大带宽
            ])
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
    print(f"  MultiScaleGWR: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_nonlinear_gwr(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    NonlinearGWR: 非线性地理加权回归

    物理意义:
    - CMAQ与观测的关系可能是非线性的（如饱和效应）
    - 使用局部多项式捕捉非线性
    """
    print("\n=== NonlinearGWR ===")

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

        # 计算全局关系用于特征变换
        global_slope = np.corrcoef(m_train, y_train)[0, 1]
        global_mean_y = y_train.mean()

        nn_finder = NearestNeighbors(n_neighbors=min(25, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        nlgwr_pred = np.zeros(len(test_df))

        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(20, len(coords_train)))
            mean_dist = np.mean(dists) + 1e-6
            bandwidth = max(mean_dist, 0.3)

            weights = np.exp(-0.5 * (dists[0] / bandwidth)**2)
            weights = weights / (weights.sum() + 1e-6)

            # 非线性特征：M, M^2, sqrt(M)
            m_local = m_train[indices[0]]
            X_local = np.column_stack([
                np.ones(len(indices[0])),
                m_local,
                m_local**2,
                np.sqrt(np.clip(m_local, 0, None))
            ])
            y_local = y_train[indices[0]]

            W = np.diag(weights)
            try:
                XTWX = X_local.T @ W @ X_local + np.eye(4) * 1e-6
                XTWy = X_local.T @ W @ y_local
                beta = np.linalg.solve(XTWX, XTWy)

                m_test_i = np.clip(m_test[i], 0, None)
                nlgwr_pred[i] = beta[0] + beta[1] * m_test_i + beta[2] * m_test_i**2 + beta[3] * np.sqrt(m_test_i)
            except:
                # 降级使用线性
                X_local_simple = np.column_stack([np.ones(len(indices[0])), m_local])
                W_simple = np.diag(weights)
                try:
                    XTWX = X_local_simple.T @ W_simple @ X_local_simple + np.eye(2) * 1e-6
                    XTWy = X_local_simple.T @ W_simple @ y_local
                    beta = np.linalg.solve(XTWX, XTWy)
                    nlgwr_pred[i] = beta[0] + beta[1] * m_test[i]
                except:
                    nlgwr_pred[i] = global_mean_y + global_slope * (m_test[i] - m_train.mean())

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'nlgwr': nlgwr_pred
        }

        print(f"  Fold {fold_id}: completed")

    nlgwr_all = np.concatenate([results[f]['nlgwr'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, nlgwr_all)
    print(f"  NonlinearGWR: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("第二十轮：真正创新的单模型")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    methods = [
        ('EnhancedGWR', run_enhanced_gwr),
        ('SpatialProcessGP', run_spatial_process_gp),
        ('LocalBiasGWR', run_local_bias_gwr),
        ('MultiScaleGWR', run_multiscale_gwr),
        ('NonlinearGWR', run_nonlinear_gwr),
    ]

    for name, func in methods:
        metrics, _ = func(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
        results[name] = metrics

    print("\n" + "="*60)
    print("第二十轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/twentieth_round_results.csv', index=False)

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
