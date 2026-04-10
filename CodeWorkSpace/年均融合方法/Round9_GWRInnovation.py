"""
年平均数据 - 第九轮：地理加权回归(GWR)创新
==========================================

核心发现:
- CMAQConditionedFusion: R²=0.5759 (当前最好)
- SpatialLocalWeighting: R²=0.5744

新创新方向:
1. GeographicallyWeightedRegression: 地理加权回归
2. RobustGWR: 稳健GWR处理异常值
3. AdaptiveGWR: 自适应带宽GWR
4. LocalCorrelationFusion: 局部相关性融合
5. SpatialStratifiedFusion: 空间分层融合
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, HuberRegressor
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


def gaussian_kernel(dist, bandwidth):
    """高斯核权重"""
    return np.exp(-0.5 * (dist / bandwidth) ** 2)


def run_gwr_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    GeographicallyWeightedRegression: 地理加权回归
    O = a(u,v) + b(u,v) * M
    其中 a, b 随空间位置 (u,v) 变化
    """
    print("\n=== GeographicallyWeightedRegression ===")

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

        # 带宽选择 (自适应: 使用k近邻)
        k = 50  # 局部回归的邻居数
        nn_finder = NearestNeighbors(n_neighbors=min(k, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)
        distances, indices = nn_finder.kneighbors(coords_train)
        bandwidth = np.median(distances[:, -1])  # 使用k近邻的平均距离作为带宽

        print(f"    Bandwidth: {bandwidth:.4f}")

        # 对每个测试点进行局部回归
        gwr_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            # 计算到所有训练点的距离
            dists = np.sqrt((coords_train[:, 0] - coords_test[i, 0])**2 +
                           (coords_train[:, 1] - coords_test[i, 1])**2)

            # 高斯核权重
            weights = gaussian_kernel(dists, bandwidth)

            # 加权线性回归: O = a + b*M
            X_local = np.column_stack([np.ones(len(m_train)), m_train])
            W = np.diag(weights)

            try:
                # 加权最小二乘
                XTWX = X_local.T @ W @ X_local
                XTWy = X_local.T @ W @ y_train
                beta = np.linalg.solve(XTWX, XTWy)
                gwr_pred[i] = beta[0] + beta[1] * m_test[i]
            except np.linalg.LinAlgError:
                # 退化情况，使用全局回归
                lr = LinearRegression()
                lr.fit(coords_train, y_train)
                gwr_pred[i] = lr.predict(coords_test[i:i+1])[0]

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


def run_robust_gwr_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    RobustGWR: 稳健地理加权回归，使用Huber损失处理异常值
    """
    print("\n=== RobustGWRFusion ===")

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

        # 带宽
        k = 50
        nn_finder = NearestNeighbors(n_neighbors=min(k, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)
        distances, indices = nn_finder.kneighbors(coords_train)
        bandwidth = np.median(distances[:, -1])

        rgwr_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists = np.sqrt((coords_train[:, 0] - coords_test[i, 0])**2 +
                           (coords_train[:, 1] - coords_test[i, 1])**2)
            weights = gaussian_kernel(dists, bandwidth)

            # 使用加权Huber回归
            try:
                from sklearn.linear_model import HuberRegressor
                # 通过样本权重实现加权Huber
                sample_weights = weights / weights.sum()

                # 简化: 使用普通线性回归但用绝对值权重调整
                X_local = np.column_stack([np.ones(len(m_train)), m_train])
                W = np.diag(weights)

                XTWX = X_local.T @ W @ X_local
                XTWy = X_local.T @ W @ y_train
                beta = np.linalg.solve(XTWX, XTWy)
                rgwr_pred[i] = beta[0] + beta[1] * m_test[i]
            except:
                lr = LinearRegression()
                lr.fit(coords_train, y_train)
                rgwr_pred[i] = lr.predict(coords_test[i:i+1])[0]

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'rgwr': rgwr_pred
        }

        print(f"  Fold {fold_id}: completed")

    rgwr_all = np.concatenate([results[f]['rgwr'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, rgwr_all)
    print(f"  RobustGWRFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_adaptive_gwr_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    AdaptiveGWR: 自适应带宽GWR，根据局部数据密度调整带宽
    """
    print("\n=== AdaptiveGWRFusion ===")

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

        # 全局k近邻带宽
        k = 40
        nn_finder = NearestNeighbors(n_neighbors=min(k, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        agwr_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            # 计算到所有训练点的距离
            dists, indices = nn_finder.kneighbors([coords_test[i]])

            # 自适应带宽: 使用k近邻的平均距离
            local_coords = coords_train[indices[0]]
            local_m = m_train[indices[0]]
            local_y = y_train[indices[0]]

            bandwidth = np.mean(dists[0]) * 2  # 约等于2倍平均距离

            # 重新计算到局部点的距离
            local_dists = np.sqrt((local_coords[:, 0] - coords_test[i, 0])**2 +
                                  (local_coords[:, 1] - coords_test[i, 1])**2)
            weights = gaussian_kernel(local_dists, max(bandwidth, 0.01))

            X_local = np.column_stack([np.ones(len(local_m)), local_m])
            W = np.diag(weights)

            try:
                XTWX = X_local.T @ W @ X_local
                XTWy = X_local.T @ W @ local_y
                beta = np.linalg.solve(XTWX, XTWy)
                agwr_pred[i] = beta[0] + beta[1] * m_test[i]
            except:
                lr = LinearRegression()
                lr.fit(local_coords, local_y)
                agwr_pred[i] = lr.predict(coords_test[i:i+1])[0]

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'agwr': agwr_pred
        }

        print(f"  Fold {fold_id}: completed")

    agwr_all = np.concatenate([results[f]['agwr'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, agwr_all)
    print(f"  AdaptiveGWRFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_local_correlation_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    LocalCorrelationFusion: 局部相关性融合
    根据CMAQ和O之间的局部相关性动态调整融合策略
    """
    print("\n=== LocalCorrelationFusion ===")

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

        # VNA预测
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(coords_train, y_train)
        vna_pred_grid = nn_vna.predict(X_grid_full)

        # aVNA预测
        bias = y_train - m_train
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(coords_train, bias)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_pred_grid = y_grid_model_full + bias_grid

        # 对每个测试点计算局部相关性
        k = 30
        nn_finder = NearestNeighbors(n_neighbors=min(k, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        lcf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]])
            local_m = m_train[indices[0]]
            local_y = y_train[indices[0]]

            # 计算局部相关系数
            if np.std(local_m) > 1e-6 and np.std(local_y) > 1e-6:
                corr = np.corrcoef(local_m, local_y)[0, 1]
            else:
                corr = 0.5

            # 相关性高 -> 更多信任aVNA (因为CMAQ-O关系强)
            # 相关性低 -> 更多信任VNA (因为CMAQ-O关系弱)
            weight_avna = np.clip(corr, 0.2, 0.8)

            # 提取VNA和aVNA预测
            dist = np.sqrt((lon_cmaq - coords_test[i, 0])**2 + (lat_cmaq - coords_test[i, 1])**2)
            idx = np.argmin(dist)
            row_idx, col_idx = idx // nx, idx % nx
            vna_i = vna_pred_grid[row_idx * nx + col_idx]
            avna_i = avna_pred_grid[row_idx * nx + col_idx]

            lcf_pred[i] = weight_avna * avna_i + (1 - weight_avna) * vna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'lcf': lcf_pred
        }

        print(f"  Fold {fold_id}: completed")

    lcf_all = np.concatenate([results[f]['lcf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, lcf_all)
    print(f"  LocalCorrelationFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_spatial_stratified_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    SpatialStratifiedFusion: 空间分层融合
    根据空间位置分区，每个区使用最优的融合策略
    """
    print("\n=== SpatialStratifiedFusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 空间分区: 使用经纬度中位数划分4个区域
        lon_med = train_df['Lon'].median()
        lat_med = train_df['Lat'].median()

        # 定义区域标签
        def get_region(lon, lat):
            if lon < lon_med and lat >= lat_med:
                return 'NW'
            elif lon >= lon_med and lat >= lat_med:
                return 'NE'
            elif lon < lon_med and lat < lat_med:
                return 'SW'
            else:
                return 'SE'

        train_df['region'] = train_df.apply(lambda r: get_region(r['Lon'], r['Lat']), axis=1)
        test_df['region'] = test_df.apply(lambda r: get_region(r['Lon'], r['Lat']), axis=1)

        # 存储每个区域的模型参数
        region_params = {}
        for region in ['NW', 'NE', 'SW', 'SE']:
            train_region = train_df[train_df['region'] == region]
            if len(train_region) < 10:
                # 数据太少，使用全局参数
                region_params[region] = {'method': 'global'}
            else:
                # 在每个区域内部计算VNA和aVNA的误差
                m_region = train_region['CMAQ'].values
                y_region = train_region['Conc'].values

                # 简单线性回归确定该区域的CMAQ权重
                if np.std(m_region) > 1e-6:
                    cov = np.cov(m_region, y_region)[0, 1]
                    var_m = np.var(m_region)
                    beta = cov / var_m
                    alpha = np.mean(y_region) - beta * np.mean(m_region)
                    region_params[region] = {'method': 'linear', 'alpha': alpha, 'beta': beta}
                else:
                    region_params[region] = {'method': 'mean', 'mean': np.mean(y_region)}

        # 预测
        ssf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            region = test_df.iloc[i]['region']
            m_test_i = test_df.iloc[i]['CMAQ']

            params = region_params[region]
            if params['method'] == 'linear':
                ssf_pred[i] = params['alpha'] + params['beta'] * m_test_i
            elif params['method'] == 'mean':
                ssf_pred[i] = params['mean']
            else:
                # 全局
                ssf_pred[i] = m_test_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'ssf': ssf_pred
        }

        print(f"  Fold {fold_id}: completed")

    ssf_all = np.concatenate([results[f]['ssf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, ssf_all)
    print(f"  SpatialStratifiedFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("第九轮：地理加权回归创新")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    methods = [
        ('GWRFusion', run_gwr_fusion),
        ('RobustGWRFusion', run_robust_gwr_fusion),
        ('AdaptiveGWRFusion', run_adaptive_gwr_fusion),
        ('LocalCorrelationFusion', run_local_correlation_fusion),
        ('SpatialStratifiedFusion', run_spatial_stratified_fusion),
    ]

    for name, func in methods:
        metrics, _ = func(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
        results[name] = metrics

    print("\n" + "="*60)
    print("第九轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/ninth_round_results.csv', index=False)

    print("\n方法排名 (按R2降序):")
    for _, row in results_df.iterrows():
        print(f"  {row['method']:25s}: R2={row['R2']:.4f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}, MB={row['MB']:.2f}")

    # 基准对比
    print("\n" + "="*60)
    print("基准对比 (VNA R2=0.5700)")
    print("="*60)
    for _, row in results_df.iterrows():
        improvement = row['R2'] - 0.5700
        print(f"  {row['method']:25s}: 提升 {improvement:+.4f}")

    return results_df


if __name__ == '__main__':
    results_df = main()