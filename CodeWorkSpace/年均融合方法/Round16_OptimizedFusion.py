"""
年平均数据 - 第十六轮：参数优化与融合增强
==========================================

当前最佳: SpatialStratifiedFusion R²=0.5787
目标: R²≥0.5800 (差距0.0013)

策略:
1. GridSearchStratifiedFusion - 网格搜索优化空间分层参数
2. DualBiasFusion - 双偏差校正融合
3. AdaptiveKNNFusion - 自适应K近邻融合
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


def run_dual_bias_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    DualBiasFusion: 双偏差校正融合

    原理:
    - 同时使用加性偏差(aVNA)和比例偏差(eVNA)
    - 根据局部偏差特征动态调整两者权重
    - 物理意义: 有些区域偏差是加性的，有些是比例的
    """
    print("\n=== DualBiasFusion ===")

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

        # 计算每个方法的预测误差
        vna_pred_train = nn_vna.predict(coords_train)
        avna_pred_train = m_train + nn_bias.predict(coords_train)
        evna_pred_train = m_train * nn_ratio.predict(coords_train)

        residual_vna = y_train - vna_pred_train
        residual_avna = y_train - avna_pred_train
        residual_evna = y_train - evna_pred_train

        std_vna = np.std(residual_vna) + 1e-6
        std_avna = np.std(residual_avna) + 1e-6
        std_evna = np.std(residual_evna) + 1e-6

        # 局部偏差特征
        k = 15
        nn_finder = NearestNeighbors(n_neighbors=min(k, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        obs_mean = y_train.mean()
        obs_std = y_train.std()

        dbf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]])

            # 局部加性偏差和比例偏差的误差比
            local_res_avna = residual_avna[indices[0]]
            local_res_evna = residual_evna[indices[0]]

            local_std_avna = np.std(local_res_avna) + 1e-6
            local_std_evna = np.std(local_res_evna) + 1e-6

            # 如果局部比例偏差误差更小，倾向于使用eVNA
            bias_type_ratio = local_std_avna / (local_std_evna + 1e-6)

            cmaq_dev = np.abs(m_test[i] - obs_mean) / obs_std

            # 基于CMAQ偏差的基础权重
            prior_vna = np.clip(0.68 - 0.32 * cmaq_dev, 0.28, 0.68)

            # 根据局部偏差类型调整aVNA/eVNA比例
            if bias_type_ratio > 1.0:
                # 加性偏差主导
                w_avna_base = 0.5
                w_evna_base = 0.5 / bias_type_ratio
            else:
                # 比例偏差主导
                w_avna_base = 0.5 * bias_type_ratio
                w_evna_base = 0.5

            # 归一化
            total_b = w_avna_base + w_evna_base
            w_avna_base /= total_b
            w_evna_base /= total_b

            # 似然权重
            like_avna = 1.0 / std_avna
            like_evna = 1.0 / std_evna

            w_vna = prior_vna
            w_avna = (1 - prior_vna) * like_avna * w_avna_base / (like_avna * w_avna_base + like_evna * w_evna_base + 1e-6)
            w_evna = (1 - prior_vna) * like_evna * w_evna_base / (like_avna * w_avna_base + like_evna * w_evna_base + 1e-6)

            total = w_vna + w_avna + w_evna
            w_vna /= total
            w_avna /= total
            w_evna /= total

            # 提取预测
            dist = np.sqrt((lon_cmaq - coords_test[i, 0])**2 + (lat_cmaq - coords_test[i, 1])**2)
            idx = np.argmin(dist)
            row_idx, col_idx = idx // nx, idx % nx
            vna_i = vna_grid[row_idx * nx + col_idx]
            avna_i = avna_grid[row_idx * nx + col_idx]
            evna_i = evna_grid[row_idx * nx + col_idx]

            dbf_pred[i] = w_vna * vna_i + w_avna * avna_i + w_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'dbf': dbf_pred
        }

        print(f"  Fold {fold_id}: completed")

    dbf_all = np.concatenate([results[f]['dbf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, dbf_all)
    print(f"  DualBiasFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_adaptive_knn_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    AdaptiveKNNFusion: 自适应K近邻融合

    原理:
    - 根据局部数据密度自适应选择K值
    - 稀疏区域用更大K，稠密区域用更小K
    - 结合多种K值的结果
    """
    print("\n=== AdaptiveKNNFusion ===")

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

        obs_mean = y_train.mean()
        obs_std = y_train.std()

        akf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            # 计算到所有训练点的距离
            dists = np.sqrt((coords_train[:, 0] - coords_test[i, 0])**2 +
                          (coords_train[:, 1] - coords_test[i, 1])**2)

            # 根据距离阈值确定近邻数量
            # 使用自适应阈值：距离小于中位数的点作为近邻
            median_dist = np.median(dists)
            adaptive_k = np.sum(dists <= median_dist * 1.5) + 1
            adaptive_k = max(10, min(30, adaptive_k))

            # 获取近邻
            indices = np.argpartition(dists, adaptive_k - 1)[:adaptive_k]
            neighbor_dists = dists[indices]

            # 距离加权
            weights = 1.0 / (neighbor_dists ** 2 + 1e-6)
            weights /= weights.sum()

            # 近邻的y和CMAQ值
            y_neighbor = y_train[indices]
            m_neighbor = m_train[indices]

            # VNA: 直接空间插值
            vna_val = np.sum(weights * y_neighbor)

            # aVNA偏差
            bias_neighbor = y_neighbor - m_neighbor
            avna_val = m_test[i] + np.sum(weights * bias_neighbor)

            # eVNA比例
            ratio_neighbor = y_neighbor / (m_neighbor + 1e-6)
            evna_val = m_test[i] * np.sum(weights * ratio_neighbor)

            # 根据CMAQ偏差确定权重
            cmaq_dev = np.abs(m_test[i] - obs_mean) / obs_std
            prior_vna = np.clip(0.68 - 0.32 * cmaq_dev, 0.28, 0.68)

            # 使用近邻残差估计不确定性
            vna_pred_neighbor = np.sum(weights * y_neighbor)
            residual_vna = y_neighbor - vna_pred_neighbor
            residual_avna = y_neighbor - (m_neighbor + np.sum(weights * bias_neighbor))
            residual_evna = y_neighbor - (m_neighbor * np.sum(weights * ratio_neighbor))

            std_vna = np.std(residual_vna) + 1e-6
            std_avna = np.std(residual_avna) + 1e-6
            std_evna = np.std(residual_evna) + 1e-6

            like_avna = 1.0 / std_avna
            like_evna = 1.0 / std_evna

            w_vna = prior_vna
            w_avna = (1 - prior_vna) * like_avna / (like_avna + like_evna)
            w_evna = (1 - prior_vna) * like_evna / (like_avna + like_evna)

            total = w_vna + w_avna + w_evna
            w_vna /= total
            w_avna /= total
            w_evna /= total

            akf_pred[i] = w_vna * vna_val + w_avna * avna_val + w_evna * evna_val

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'akf': akf_pred
        }

        print(f"  Fold {fold_id}: completed")

    akf_all = np.concatenate([results[f]['akf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, akf_all)
    print(f"  AdaptiveKNNFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_optimized_spatial_stratified(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    OptimizedSpatialStratifiedFusion: 优化空间分层融合

    基于SpatialStratifiedFusion但优化参数
    """
    print("\n=== OptimizedSpatialStratifiedFusion ===")

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

        # 局部偏差
        k = 15
        nn_finder = NearestNeighbors(n_neighbors=min(k, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        obs_mean = y_train.mean()
        obs_std = y_train.std()

        ossf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]])

            # 连续局部偏差强度
            local_vna = nn_vna.predict(coords_train[indices[0]])
            local_bias = np.mean(y_train[indices[0]] - local_vna)
            local_bias_norm = np.clip(local_bias / obs_std, -1, 1)

            cmaq_dev = np.abs(m_test[i] - obs_mean) / obs_std

            # 优化后的权重公式
            bias_adjustment = 0.18 * local_bias_norm
            prior_vna_base = 0.70 - 0.35 * cmaq_dev
            prior_vna = np.clip(prior_vna_base + bias_adjustment, 0.20, 0.72)

            like_avna = 1.0 / std_avna
            like_evna = 1.0 / std_evna

            w_vna = prior_vna
            w_avna = (1 - prior_vna) * like_avna / (like_avna + like_evna)
            w_evna = (1 - prior_vna) * like_evna / (like_avna + like_evna)

            total = w_vna + w_avna + w_evna
            w_vna /= total
            w_avna /= total
            w_evna /= total

            # 提取预测
            dist = np.sqrt((lon_cmaq - coords_test[i, 0])**2 + (lat_cmaq - coords_test[i, 1])**2)
            idx = np.argmin(dist)
            row_idx, col_idx = idx // nx, idx % nx
            vna_i = vna_grid[row_idx * nx + col_idx]
            avna_i = avna_grid[row_idx * nx + col_idx]
            evna_i = evna_grid[row_idx * nx + col_idx]

            ossf_pred[i] = w_vna * vna_i + w_avna * avna_i + w_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'ossf': ossf_pred
        }

        print(f"  Fold {fold_id}: completed")

    ossf_all = np.concatenate([results[f]['ossf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, ossf_all)
    print(f"  OptimizedSpatialStratifiedFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("第十六轮：参数优化与融合增强")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    methods = [
        ('DualBiasFusion', run_dual_bias_fusion),
        ('AdaptiveKNNFusion', run_adaptive_knn_fusion),
        ('OptimizedSpatialStratifiedFusion', run_optimized_spatial_stratified),
    ]

    for name, func in methods:
        metrics, _ = func(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
        results[name] = metrics

    print("\n" + "="*60)
    print("第十六轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/sixteenth_round_results.csv', index=False)

    print("\n方法排名 (按R2降序):")
    for _, row in results_df.iterrows():
        print(f"  {row['method']:35s}: R2={row['R2']:.4f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}, MB={row['MB']:.2f}")

    # 基准对比
    print("\n" + "="*60)
    print("基准对比 (VNA R2=0.5700, 目标 R2>=0.5800)")
    print("="*60)
    for _, row in results_df.iterrows():
        improvement = row['R2'] - 0.5700
        status = "达标" if improvement >= 0.01 else "未达标"
        print(f"  {row['method']:35s}: 提升 {improvement:+.4f} [{status}]")

    # 与之前最佳对比
    print("\n" + "="*60)
    print("与之前最佳对比 (SpatialStratifiedFusion R2=0.5787)")
    print("="*60)
    for _, row in results_df.iterrows():
        diff = row['R2'] - 0.5787
        print(f"  {row['method']:35s}: 差距 {diff:+.4f}")

    return results_df


if __name__ == '__main__':
    results_df = main()