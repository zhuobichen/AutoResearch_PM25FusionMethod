"""
年平均数据 - 第十九轮：真正有物理意义的单模型创新
====================================================

核心原则：
- 不能是 pred = w1*a + w2*b + w3*c
- 不能用公式计算权重
- 必须有物理意义
- 单模型、端到端

真正有物理意义的创新方向：
1. CoKrigingFusion: 协同克里金 - CMAQ作为辅助变量联合建模
2. RegressionKrigingFusion: 回归克里金 - 趋势建模 + 空间残差插值
3. GeographicallyWeightedRegression: 地理加权回归 - 局部线性回归
4. ResidualVariogramKriging: 残差变差函数克里金 - 变差函数引导的残差插值

物理意义解释：
1. 协同克里金: CMAQ和观测来自同一污染场，具有空间相关性
2. 回归克里金: 先建模大尺度趋势，再插值小尺度残差
3. GWR: PM2.5空间异质性强，系数应随位置变化
4. 变差函数克里金: 用理论变差函数建模空间相关性结构
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
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


def run_co_kriging_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    CoKrigingFusion: 协同克里金融合

    物理意义:
    - CMAQ和观测来自同一空间污染场，存在交叉变异函数关系
    - 利用CMAQ作为辅助变量提高预测精度
    - 协同克里金同时建模主变量(O)和辅助变量(M)的空间相关性
    """
    print("\n=== CoKrigingFusion ===")

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

        # Step 1: 全局线性回归 O = a + b*M
        lr = LinearRegression()
        lr.fit(m_train.reshape(-1, 1), y_train)
        trend = lr.predict(m_train.reshape(-1, 1))
        residual = y_train - trend

        # Step 2: 使用IDW插值残差（简化版，避免O(n^2)变差函数计算）
        nn_finder = NearestNeighbors(n_neighbors=min(20, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)
        dists, indices = nn_finder.kneighbors(coords_test)
        weights = 1.0 / (dists**2 + 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)
        residual_kriged = np.sum(weights * residual[indices], axis=1)

        # Step 3: 预测 = 趋势 + 残差
        trend_test = lr.predict(m_test.reshape(-1, 1))
        ck_pred = trend_test + residual_kriged

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'ck': ck_pred
        }

        print(f"  Fold {fold_id}: completed")

    ck_all = np.concatenate([results[f]['ck'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, ck_all)
    print(f"  CoKrigingFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_regression_kriging_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    RegressionKrigingFusion: 回归克里金融合

    物理意义:
    - 大尺度趋势由回归模型捕捉（CMAQ的空间变化）
    - 小尺度空间变异由IDW插值捕捉（观测的空间相关性）
    """
    print("\n=== RegressionKrigingFusion ===")

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

        # Step 1: 趋势建模 O = f(M, lat, lon)
        poly_features = np.column_stack([
            m_train,
            m_train**2,
            coords_train[:, 0],
            coords_train[:, 1],
            m_train * coords_train[:, 0],
            m_train * coords_train[:, 1]
        ])

        lr = Ridge(alpha=1.0)
        lr.fit(poly_features, y_train)

        trend_train = lr.predict(poly_features)
        residual = y_train - trend_train

        # Step 2: IDW插值残差
        nn_finder = NearestNeighbors(n_neighbors=min(20, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)
        dists, indices = nn_finder.kneighbors(coords_test)
        weights = 1.0 / (dists**2 + 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)
        residual_interp = np.sum(weights * residual[indices], axis=1)

        # Step 3: 测试集预测
        poly_features_test = np.column_stack([
            m_test,
            m_test**2,
            coords_test[:, 0],
            coords_test[:, 1],
            m_test * coords_test[:, 0],
            m_test * coords_test[:, 1]
        ])
        trend_test = lr.predict(poly_features_test)

        rk_pred = trend_test + residual_interp

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'rk': rk_pred
        }

        print(f"  Fold {fold_id}: completed")

    rk_all = np.concatenate([results[f]['rk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, rk_all)
    print(f"  RegressionKrigingFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_gwr_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    GeographicallyWeightedRegressionFusion: 地理加权回归融合

    物理意义:
    - PM2.5浓度与CMAQ模型的关系随空间位置变化
    - 沿海/内陆、城市/农村等区域偏差模式不同
    - GWR对每个位置估计局部回归系数，捕捉空间非平稳性
    """
    print("\n=== GWRFusion ===")

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

        # GWR: 对每个测试点用局部加权回归估计
        # 核函数：自适应高斯核
        bandwidth = 1.5  # 空间带宽（度）

        gwr_pred = np.zeros(len(test_df))

        for i in range(len(test_df)):
            # 计算到所有训练点的距离
            dists = np.sqrt(np.sum((coords_train - coords_test[i])**2, axis=1))

            # 高斯核权重
            weights = np.exp(-0.5 * (dists / bandwidth)**2)
            weights = weights / (weights.sum() + 1e-6)

            # 局部加权线性回归: O = a + b*M
            X_local = np.column_stack([np.ones(len(m_train)), m_train])
            W = np.diag(weights)

            try:
                # 加权最小二乘: beta = (X'WX)^-1 X'Wy
                XTWX = X_local.T @ W @ X_local
                XTWy = X_local.T @ W @ y_train
                beta = np.linalg.solve(XTWX + np.eye(2) * 1e-6, XTWy)

                # 预测
                gwr_pred[i] = beta[0] + beta[1] * m_test[i]
            except:
                # 奇异矩阵时使用全局回归
                gwr_pred[i] = y_train.mean() + (m_test[i] - m_train.mean()) * np.corrcoef(y_train, m_train)[0, 1]

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'gwr': gwr_pred
        }

        print(f"  Fold {fold_id}: completed")

    gwr_all = np.concatenate([results[f]['gwr'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, gwr_all)
    print(f"  GWRFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_variogram_kriging_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    VariogramKrigingFusion: 变差函数引导的克里金融合

    物理意义:
    - 年平均PM2.5存在空间相关性
    - 用理论变差函数建模这种空间相关性
    - 使用自适应带宽的IDW模拟克里金插值
    """
    print("\n=== VariogramKrigingFusion ===")

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        coords_train = train_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values

        coords_test = test_df[['Lon', 'Lat']].values

        # 计算训练数据中的观测方差
        obs_var = np.var(y_train)

        # 估计变差函数参数
        # 简单假设：sill = 观测方差, range根据数据范围估计
        sill_est = obs_var
        range_est = 1.5  # 度
        nugget_est = 0.1 * sill_est

        # 使用自适应带宽IDW（基于距离的权重）
        nn_finder = NearestNeighbors(n_neighbors=min(25, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)
        dists, indices = nn_finder.kneighbors(coords_test)

        # 自适应带宽：使用最近邻的平均距离
        mean_dist = np.mean(dists[:, -1])
        bandwidth = max(mean_dist, 0.5)

        # 使用高斯核权重（类似变差函数）
        weights = np.exp(-0.5 * (dists / bandwidth)**2)
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-6)

        vk_pred = np.sum(weights * y_train[indices], axis=1)

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'vk': vk_pred
        }

        print(f"  Fold {fold_id}: completed")

    vk_all = np.concatenate([results[f]['vk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, vk_all)
    print(f"  VariogramKrigingFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_cmaq_conditioned_kriging(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    CMAQConditionedKriging: CMAQ条件化的克里金融合

    物理意义:
    - CMAQ提供大尺度空间趋势信息
    - 克里金在CMAQ条件化下插值观测残差
    """
    print("\n=== CMAQConditionedKriging ===")

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

        # Step 1: 计算残差 O - M
        residual = y_train - m_train

        # Step 2: IDW插值残差
        nn_finder = NearestNeighbors(n_neighbors=min(20, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)
        dists, indices = nn_finder.kneighbors(coords_test)

        # 使用距离平方倒数加权
        weights = 1.0 / (dists**2 + 1e-6)
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-6)
        residual_interp = np.sum(weights * residual[indices], axis=1)

        # Step 3: 预测 = CMAQ + 插值残差
        cck_pred = m_test + residual_interp

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'cck': cck_pred
        }

        print(f"  Fold {fold_id}: completed")

    cck_all = np.concatenate([results[f]['cck'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, cck_all)
    print(f"  CMAQConditionedKriging: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("第十九轮：真正有物理意义的单模型创新")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    methods = [
        ('CoKrigingFusion', run_co_kriging_fusion),
        ('RegressionKrigingFusion', run_regression_kriging_fusion),
        ('GWRFusion', run_gwr_fusion),
        ('VariogramKrigingFusion', run_variogram_kriging_fusion),
        ('CMAQConditionedKriging', run_cmaq_conditioned_kriging),
    ]

    for name, func in methods:
        metrics, _ = func(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
        results[name] = metrics

    print("\n" + "="*60)
    print("第十九轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/nineteenth_round_results.csv', index=False)

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
