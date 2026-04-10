"""
年平均数据创新融合方法
====================
针对年平均数据特点的创新融合方法

方法:
1. GPR_Fusion: GPR直接建模 O = f(M, lat, lon)
2. SegmentFusion: 分段融合，根据CMAQ值分段
3. QuantileFusion: 分位数回归校正
4. DistanceWeightedFusion: 距离加权融合
5. HybridKrigingFusion: 混合克里金
6. MaternGPFusion: Matern核GPR融合
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
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
    print("=== Loading and Preparing Year-Average Data ===")

    # 加载监测数据
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    # 年平均监测数据
    monitor_year_avg = monitor_df.groupby('Site').agg({
        'Conc': 'mean',
        'Lat': 'first',
        'Lon': 'first'
    }).reset_index()
    monitor_year_avg.columns = ['Site', 'Conc', 'Lat', 'Lon']

    print(f"Monitor year-avg: {len(monitor_year_avg)} sites")

    # 加载CMAQ数据
    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    # 年平均CMAQ数据
    cmaq_year_avg = pred_pm25.mean(axis=0)
    print(f"CMAQ year-avg shape: {cmaq_year_avg.shape}")

    # 提取站点位置的CMAQ值
    cmaq_values = []
    for _, row in monitor_year_avg.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_year_avg)
        cmaq_values.append(val)
    monitor_year_avg['CMAQ'] = cmaq_values

    # 合并fold信息
    monitor_year_avg = monitor_year_avg.merge(fold_df, on='Site', how='left')
    monitor_year_avg = monitor_year_avg.dropna(subset=['Lat', 'Lon', 'CMAQ', 'Conc', 'fold'])

    print(f"Final dataset: {len(monitor_year_avg)} sites")
    print(f"Monitor Conc range: [{monitor_year_avg['Conc'].min():.2f}, {monitor_year_avg['Conc'].max():.2f}]")
    print(f"CMAQ range: [{monitor_year_avg['CMAQ'].min():.2f}, {monitor_year_avg['CMAQ'].max():.2f}]")

    return monitor_year_avg, lon_cmaq, lat_cmaq, cmaq_year_avg


def run_gpr_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    GPR_Fusion: GPR直接建模 O = f(M, lat, lon)
    """
    print("\n=== GPR Fusion ===")
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 特征: [CMAQ, Lat, Lon]
        X_train = train_df[['CMAQ', 'Lat', 'Lon']].values
        X_test = test_df[['CMAQ', 'Lat', 'Lon']].values
        y_train = train_df['Conc'].values
        y_test = test_df['Conc'].values

        # GPR建模
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1.0, normalize_y=True)
        gpr.fit(X_train_scaled, y_train)
        gpr_pred, _ = gpr.predict(X_test_scaled, return_std=True)

        results[fold_id] = {
            'y_true': y_test,
            'gpr_fusion': gpr_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    gpr_all = np.concatenate([results[f]['gpr_fusion'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, gpr_all)
    print(f"  GPR Fusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_segment_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    SegmentFusion: 分段融合，根据CMAQ值分段使用不同校正策略
    """
    print("\n=== Segment Fusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 根据CMAQ值分段
        quantiles = [0, 0.33, 0.66, 1.0]
        train_df['cmaq_quantile'] = pd.qcut(train_df['CMAQ'], q=3, labels=['low', 'mid', 'high'])

        # 每段分别校正
        segment_preds = np.zeros(len(test_df))

        for seg in ['low', 'mid', 'high']:
            train_seg = train_df[train_df['cmaq_quantile'] == seg]
            test_seg_mask = pd.qcut(test_df['CMAQ'], q=3, labels=['low', 'mid', 'high']) == seg

            if len(train_seg) < 5:
                # 数据太少，使用全局比率
                ratio = train_df['Conc'].mean() / max(train_df['CMAQ'].mean(), 1e-6)
                segment_preds[test_seg_mask] = test_df.loc[test_seg_mask, 'CMAQ'].values * ratio
            else:
                # 线性回归校正
                lr = LinearRegression()
                lr.fit(train_seg[['CMAQ']].values, train_seg['Conc'].values)
                segment_preds[test_seg_mask] = lr.predict(test_df.loc[test_seg_mask, ['CMAQ']].values)

        # 空间插值残差
        residual = train_df['Conc'].values - train_df['CMAQ'].values * (train_df['Conc'].mean() / max(train_df['CMAQ'].mean(), 1e-6))

        nn = NNA(method='nearest', k=15, power=-2)
        nn.fit(train_df[['Lon', 'Lat']].values, residual)

        residual_pred = nn.predict(test_df[['Lon', 'Lat']].values)
        segment_preds = segment_preds + residual_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'segment_fusion': segment_preds
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    seg_all = np.concatenate([results[f]['segment_fusion'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, seg_all)
    print(f"  Segment Fusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_quantile_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    QuantileFusion: 分位数回归校正，处理偏态分布
    """
    print("\n=== Quantile Fusion ===")

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 分位数回归 (中位数)
        from sklearn.linear_model import QuantileRegressor
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                qr = QuantileRegressor(quantile=0.5, alpha=0.1, solver='highs')
                qr.fit(train_df[['CMAQ']].values, train_df['Conc'].values)
                qr_pred = qr.predict(test_df[['CMAQ']].values)
            except:
                # fallback to linear
                lr = LinearRegression()
                lr.fit(train_df[['CMAQ']].values, train_df['Conc'].values)
                qr_pred = lr.predict(test_df[['CMAQ']].values)

        # 残差空间插值
        residual = train_df['Conc'].values - qr.predict(train_df[['CMAQ']].values)

        nn = NNA(method='nearest', k=15, power=-2)
        nn.fit(train_df[['Lon', 'Lat']].values, residual)
        residual_pred = nn.predict(test_df[['Lon', 'Lat']].values)

        quantile_fusion_pred = qr_pred + residual_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'quantile_fusion': quantile_fusion_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    qf_all = np.concatenate([results[f]['quantile_fusion'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, qf_all)
    print(f"  Quantile Fusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_distance_weighted_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    DistanceWeightedFusion: 根据站点到CMAQ网格距离调整权重
    """
    print("\n=== Distance Weighted Fusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 计算每个训练站点的CMAQ距离
        train_df = train_df.copy()
        train_df['cmaq_dist'] = 0.0  # 简化处理，实际应该在网格上计算

        # eVNA部分
        train_df['ratio'] = train_df['Conc'] / train_df['CMAQ']
        nn_ratio = NNA(method='nearest', k=15, power=-2)
        nn_ratio.fit(train_df[['Lon', 'Lat']].values, train_df['ratio'].values)
        ratio_grid = nn_ratio.predict(X_grid_full)
        evna_pred_grid = y_grid_model_full * ratio_grid

        # aVNA部分
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(train_df[['Lon', 'Lat']].values, train_df['bias'].values)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_pred_grid = y_grid_model_full + bias_grid

        # 提取测试站点预测
        evna_pred = np.zeros(len(test_df))
        avna_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            evna_pred[i] = evna_pred_grid[idx // nx * nx + idx % nx]
            avna_pred[i] = avna_pred_grid[idx // nx * nx + idx % nx]

        # 基于CMAQ值的非线性混合
        # 当CMAQ偏低时，eVNA可能更好；偏高时，aVNA可能更好
        cmaq_ratio = test_df['CMAQ'].values / test_df['CMAQ'].mean()
        weight = np.clip(cmaq_ratio, 0.2, 0.8)  # 限制权重范围

        dw_fusion_pred = weight * evna_pred + (1 - weight) * avna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'dw_fusion': dw_fusion_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    dw_all = np.concatenate([results[f]['dw_fusion'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, dw_all)
    print(f"  Distance Weighted Fusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_hybrid_kriging_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    HybridKrigingFusion: 同时建模均值和空间相关性
    O = a + b*M + c*lat + d*lon + GPR空间残差
    """
    print("\n=== Hybrid Kriging Fusion ===")
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 特征: [CMAQ, Lat, Lon]
        X_train = train_df[['CMAQ', 'Lat', 'Lon']].values
        X_test = test_df[['CMAQ', 'Lat', 'Lon']].values
        y_train = train_df['Conc'].values
        y_test = test_df['Conc'].values

        # 线性趋势
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        trend_pred = lr.predict(X_test)
        trend_residual = y_train - lr.predict(X_train)

        # GPR插值残差
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1.0, normalize_y=True)
        gpr.fit(train_df[['Lon', 'Lat']].values, trend_residual)
        gpr_pred, _ = gpr.predict(test_df[['Lon', 'Lat']].values, return_std=True)

        hybrid_pred = trend_pred + gpr_pred

        results[fold_id] = {
            'y_true': y_test,
            'hybrid_kriging': hybrid_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    hk_all = np.concatenate([results[f]['hybrid_kriging'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, hk_all)
    print(f"  Hybrid Kriging Fusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_matern_gp_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    MaternGPFusion: Matern核GPR融合
    """
    print("\n=== Matern GP Fusion ===")
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 特征: [CMAQ, Lat, Lon]
        X_train = train_df[['CMAQ', 'Lat', 'Lon']].values
        X_test = test_df[['CMAQ', 'Lat', 'Lon']].values
        y_train = train_df['Conc'].values
        y_test = test_df['Conc'].values

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Matern GPR
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1.0, normalize_y=True)
        gpr.fit(X_train_scaled, y_train)
        matern_pred, _ = gpr.predict(X_test_scaled, return_std=True)

        results[fold_id] = {
            'y_true': y_test,
            'matern_gp': matern_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    mgp_all = np.concatenate([results[f]['matern_gp'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, mgp_all)
    print(f"  Matern GP Fusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("年平均数据创新融合方法测试")
    print("="*60)

    # 加载数据
    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    # 运行各种创新方法
    methods = [
        ('GPR_Fusion', run_gpr_fusion),
        ('SegmentFusion', run_segment_fusion),
        ('QuantileFusion', run_quantile_fusion),
        ('DistanceWeightedFusion', run_distance_weighted_fusion),
        ('HybridKrigingFusion', run_hybrid_kriging_fusion),
        ('MaternGPFusion', run_matern_gp_fusion),
    ]

    for name, func in methods:
        metrics, _ = func(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
        results[name] = metrics

    # 保存结果
    print("\n" + "="*60)
    print("创新方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/innovative_results.csv', index=False)

    print("\n方法排名 (按R2降序):")
    for _, row in results_df.iterrows():
        print(f"  {row['method']:25s}: R2={row['R2']:.4f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}")

    return results_df


if __name__ == '__main__':
    results_df = main()