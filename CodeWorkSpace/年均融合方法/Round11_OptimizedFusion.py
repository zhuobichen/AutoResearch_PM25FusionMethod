"""
年平均数据 - 第十一轮：优化KernelAwareFusion与新创新
====================================================

当前最佳:
- KernelAwareFusion: R²=0.5770 (提升 +0.0070)
- CMAQConditionedFusion: R²=0.5759
- CMAQGuidedFusion: R²=0.5756

新方法:
1. OptimizedKernelAwareFusion: 优化KernelAwareFusion参数
2. TripleBandwidthFusion: 三带宽融合
3. CorrelationGuidedFusion: 相关性引导融合
4. BayesianInspiredFusion: 贝叶斯启发的融合
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


def run_optimized_kernel_aware_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    OptimizedKernelAwareFusion: 优化KernelAwareFusion
    使用不同的k值和power参数组合
    """
    print("\n=== OptimizedKernelAwareFusion ===")

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

        # eVNA with k=20, power=-2.5
        ratio = y_train / m_train
        nn_ratio = NNA(method='nearest', k=20, power=-2.5)
        nn_ratio.fit(coords_train, ratio)
        ratio_grid = nn_ratio.predict(X_grid_full)
        evna_grid = y_grid_model_full * ratio_grid

        # aVNA with k=15, power=-2
        bias = y_train - m_train
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(coords_train, bias)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_grid = y_grid_model_full + bias_grid

        # VNA with k=15, power=-2
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(coords_train, y_train)
        vna_grid = nn_vna.predict(X_grid_full)

        # 优化权重参数
        obs_mean = y_train.mean()
        obs_std = y_train.std()

        okaf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            cmaq_dev = (m_test[i] - obs_mean) / obs_std

            # 优化后的权重公式
            w_vna = np.clip(0.70 - 0.35 * np.abs(cmaq_dev), 0.25, 0.70)
            w_avna = np.clip(0.15 + 0.30 * np.abs(cmaq_dev), 0.15, 0.50)
            w_evna = 1 - w_vna - w_avna

            dist = np.sqrt((lon_cmaq - coords_test[i, 0])**2 + (lat_cmaq - coords_test[i, 1])**2)
            idx = np.argmin(dist)
            row_idx, col_idx = idx // nx, idx % nx
            vna_i = vna_grid[row_idx * nx + col_idx]
            avna_i = avna_grid[row_idx * nx + col_idx]
            evna_i = evna_grid[row_idx * nx + col_idx]

            okaf_pred[i] = w_vna * vna_i + w_avna * avna_i + w_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'okaf': okaf_pred
        }

        print(f"  Fold {fold_id}: completed")

    okaf_all = np.concatenate([results[f]['okaf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, okaf_all)
    print(f"  OptimizedKernelAwareFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_triple_bandwidth_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    TripleBandwidthFusion: 三带宽融合
    使用不同的k值捕获不同尺度的空间信息
    """
    print("\n=== TripleBandwidthFusion ===")

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

        # 多尺度VNA
        vna_preds = []
        for k, w in [(8, 0.25), (15, 0.50), (30, 0.25)]:
            nn = NNA(method='nearest', k=k, power=-2)
            nn.fit(coords_train, y_train)
            pred_grid = nn.predict(X_grid_full)
            pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, pred_grid, ny, nx)
            vna_preds.append(w * pred)
        vna_pred = np.sum(vna_preds, axis=0)

        # 多尺度aVNA
        bias = y_train - m_train
        avna_preds = []
        for k, w in [(8, 0.25), (15, 0.50), (30, 0.25)]:
            nn = NNA(method='nearest', k=k, power=-2)
            nn.fit(coords_train, bias)
            pred_grid = nn.predict(X_grid_full) + y_grid_model_full
            pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, pred_grid, ny, nx)
            avna_preds.append(w * pred)
        avna_pred = np.sum(avna_preds, axis=0)

        # 多尺度eVNA
        ratio = y_train / m_train
        evna_preds = []
        for k, w in [(8, 0.25), (15, 0.50), (30, 0.25)]:
            nn = NNA(method='nearest', k=k, power=-2)
            nn.fit(coords_train, ratio)
            pred_grid = nn.predict(X_grid_full) * y_grid_model_full
            pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, pred_grid, ny, nx)
            evna_preds.append(w * pred)
        evna_pred = np.sum(evna_preds, axis=0)

        # 动态权重
        obs_mean = y_train.mean()
        obs_std = y_train.std()
        cmaq_dev = np.abs(m_test - obs_mean) / obs_std

        w_vna = np.clip(0.60 - 0.30 * cmaq_dev, 0.25, 0.65)
        w_avna = np.clip(0.20 + 0.25 * cmaq_dev, 0.20, 0.50)
        w_evna = 1 - w_vna - w_avna

        tbf_pred = w_vna * vna_pred + w_avna * avna_pred + w_evna * evna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'tbf': tbf_pred
        }

        print(f"  Fold {fold_id}: completed")

    tbf_all = np.concatenate([results[f]['tbf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, tbf_all)
    print(f"  TripleBandwidthFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_correlation_guided_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    CorrelationGuidedFusion: 相关性引导融合
    根据CMAQ-O相关性调整融合权重
    """
    print("\n=== CorrelationGuidedFusion ===")

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

        # VNA
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(coords_train, y_train)
        vna_grid = nn_vna.predict(X_grid_full)

        # aVNA
        bias = y_train - m_train
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(coords_train, bias)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_grid = y_grid_model_full + bias_grid

        # eVNA
        ratio = y_train / m_train
        nn_ratio = NNA(method='nearest', k=15, power=-2)
        nn_ratio.fit(coords_train, ratio)
        ratio_grid = nn_ratio.predict(X_grid_full)
        evna_grid = y_grid_model_full * ratio_grid

        # 计算全局和局部相关性
        global_corr = np.corrcoef(m_train, y_train)[0, 1]

        # 局部相关性
        k = 20
        nn_finder = NearestNeighbors(n_neighbors=min(k, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        cgf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]])
            local_m = m_train[indices[0]]
            local_y = y_train[indices[0]]

            if np.std(local_m) > 1e-6 and np.std(local_y) > 1e-6:
                local_corr = np.corrcoef(local_m, local_y)[0, 1]
            else:
                local_corr = global_corr

            # 相关性引导权重
            # 相关性高 -> 更多信任aVNA/eVNA
            # 相关性低 -> 更多信任VNA
            avg_corr = 0.5 * (global_corr + local_corr)
            weight_vna = np.clip(0.70 - 0.40 * avg_corr, 0.25, 0.65)
            weight_avna = np.clip(0.15 + 0.25 * avg_corr, 0.15, 0.45)
            weight_evna = 1 - weight_vna - weight_avna

            dist = np.sqrt((lon_cmaq - coords_test[i, 0])**2 + (lat_cmaq - coords_test[i, 1])**2)
            idx = np.argmin(dist)
            row_idx, col_idx = idx // nx, idx % nx
            vna_i = vna_grid[row_idx * nx + col_idx]
            avna_i = avna_grid[row_idx * nx + col_idx]
            evna_i = evna_grid[row_idx * nx + col_idx]

            cgf_pred[i] = weight_vna * vna_i + weight_avna * avna_i + weight_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'cgf': cgf_pred
        }

        print(f"  Fold {fold_id}: completed")

    cgf_all = np.concatenate([results[f]['cgf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, cgf_all)
    print(f"  CorrelationGuidedFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_bayesian_inspired_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    BayesianInspiredFusion: 贝叶斯启发的融合
    使用贝叶斯思路: 结合先验(VNA)和似然(CMAQ校正)
    """
    print("\n=== BayesianInspiredFusion ===")

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

        # VNA (先验)
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(coords_train, y_train)
        vna_grid = nn_vna.predict(X_grid_full)

        # aVNA (似然1)
        bias = y_train - m_train
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(coords_train, bias)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_grid = y_grid_model_full + bias_grid

        # eVNA (似然2)
        ratio = y_train / m_train
        nn_ratio = NNA(method='nearest', k=15, power=-2)
        nn_ratio.fit(coords_train, ratio)
        ratio_grid = nn_ratio.predict(X_grid_full)
        evna_grid = y_grid_model_full * ratio_grid

        # 贝叶斯权重: P(VNA|data) ∝ P(data|VNA) * P(VNA)
        obs_mean = y_train.mean()
        obs_std = y_train.std()

        # 计算每个模型的不确定性(用残差标准差)
        vna_residual = y_train - nn_vna.predict(coords_train)
        avna_pred_train = m_train + nn_bias.predict(coords_train)
        avna_residual = y_train - avna_pred_train
        evna_pred_train = m_train * nn_ratio.predict(coords_train)
        evna_residual = y_train - evna_pred_train

        std_vna = np.std(vna_residual) + 1e-6
        std_avna = np.std(avna_residual) + 1e-6
        std_evna = np.std(evna_residual) + 1e-6

        bif_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            # 先验权重 (基于CMAQ偏差)
            cmaq_dev = np.abs(m_test[i] - obs_mean) / obs_std
            prior_vna = np.clip(0.65 - 0.30 * cmaq_dev, 0.25, 0.65)

            # 似然权重 (基于不确定性)
            # 更小的残差std意味着更高的似然
            like_avna = 1.0 / std_avna
            like_evna = 1.0 / std_evna

            # 后验权重 (近似贝叶斯更新)
            w_vna = prior_vna
            w_avna = (1 - prior_vna) * like_avna / (like_avna + like_evna + 1e-6)
            w_evna = (1 - prior_vna) * like_evna / (like_avna + like_evna + 1e-6)

            # 归一化
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

            bif_pred[i] = w_vna * vna_i + w_avna * avna_i + w_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'bif': bif_pred
        }

        print(f"  Fold {fold_id}: completed")

    bif_all = np.concatenate([results[f]['bif'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, bif_all)
    print(f"  BayesianInspiredFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("第十一轮：优化KernelAwareFusion与新创新")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    methods = [
        ('OptimizedKernelAwareFusion', run_optimized_kernel_aware_fusion),
        ('TripleBandwidthFusion', run_triple_bandwidth_fusion),
        ('CorrelationGuidedFusion', run_correlation_guided_fusion),
        ('BayesianInspiredFusion', run_bayesian_inspired_fusion),
    ]

    for name, func in methods:
        metrics, _ = func(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
        results[name] = metrics

    print("\n" + "="*60)
    print("第十一轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/eleventh_round_results.csv', index=False)

    print("\n方法排名 (按R2降序):")
    for _, row in results_df.iterrows():
        print(f"  {row['method']:30s}: R2={row['R2']:.4f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}, MB={row['MB']:.2f}")

    # 基准对比
    print("\n" + "="*60)
    print("基准对比 (VNA R2=0.5700)")
    print("="*60)
    for _, row in results_df.iterrows():
        improvement = row['R2'] - 0.5700
        print(f"  {row['method']:30s}: 提升 {improvement:+.4f}")

    return results_df


if __name__ == '__main__':
    results_df = main()