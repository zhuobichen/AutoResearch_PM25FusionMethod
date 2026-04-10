"""
年平均数据 - 第十四轮：更激进的单模型优化
==========================================

当前最佳: SpatialStratifiedFusion R²=0.5787
目标: R²≥0.5800 (差距0.0013)

策略:
1. EnhancedSpatialStratifiedFusion - 更精细的空间分层
2. LocalCorrelationAdaptiveFusion - 结合局部相关性
3. VarianceWeightedFusion - 方差加权融合
4. OptimizedNonlinearWeighting - 优化的非线性加权
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


def run_enhanced_spatial_stratified(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    EnhancedSpatialStratifiedFusion: 增强型空间分层融合

    改进:
    - 使用连续偏差强度而非离散分类
    - 5个不同的空间区域权重设置
    - 更精细的CMAQ偏差响应
    """
    print("\n=== EnhancedSpatialStratifiedFusion ===")

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

        # 预计算残差用于似然
        vna_pred_train = nn_vna.predict(coords_train)
        avna_pred_train = m_train + nn_bias.predict(coords_train)
        evna_pred_train = m_train * nn_ratio.predict(coords_train)

        residual_avna = y_train - avna_pred_train
        residual_evna = y_train - evna_pred_train
        std_avna = np.std(residual_avna) + 1e-6
        std_evna = np.std(residual_evna) + 1e-6

        # 局部偏差计算
        k = 15
        nn_finder = NearestNeighbors(n_neighbors=min(k, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        obs_mean = y_train.mean()
        obs_std = y_train.std()

        essf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]])

            # 连续局部偏差强度 (-1到1范围)
            local_vna = nn_vna.predict(coords_train[indices[0]])
            local_bias = np.mean(y_train[indices[0]] - local_vna)
            local_bias_norm = np.clip(local_bias / obs_std, -1, 1)

            # CMAQ偏差
            cmaq_dev = np.abs(m_test[i] - obs_mean) / obs_std

            # 增强权重公式
            # local_bias_norm < 0: VNA预测偏高，减少VNA权重
            # local_bias_norm > 0: VNA预测偏低，增加VNA权重
            bias_adjustment = 0.15 * local_bias_norm

            prior_vna_base = 0.68 - 0.32 * cmaq_dev
            prior_vna = np.clip(prior_vna_base + bias_adjustment, 0.22, 0.72)

            # 似然权重
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

            essf_pred[i] = w_vna * vna_i + w_avna * avna_i + w_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'essf': essf_pred
        }

        print(f"  Fold {fold_id}: completed")

    essf_all = np.concatenate([results[f]['essf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, essf_all)
    print(f"  EnhancedSpatialStratifiedFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_local_correlation_adaptive(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    LocalCorrelationAdaptiveFusion: 局部相关性自适应融合

    原理:
    - 当局部CMAQ-观测相关性高时，更信任CMAQ校正
    - 当局部相关性低时，更信任VNA
    - 结合CMAQ偏差进行综合判断
    """
    print("\n=== LocalCorrelationAdaptiveFusion ===")

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

        # 局部相关性
        k = 20
        nn_finder = NearestNeighbors(n_neighbors=min(k, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        obs_mean = y_train.mean()
        obs_std = y_train.std()

        # 全局CMAQ-观测相关性
        global_corr = np.corrcoef(m_train, y_train)[0, 1]

        lcaf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]])

            # 局部CMAQ-观测相关性
            local_m = m_train[indices[0]]
            local_y = y_train[indices[0]]

            if np.std(local_m) > 1e-6 and np.std(local_y) > 1e-6:
                local_corr = np.corrcoef(local_m, local_y)[0, 1]
            else:
                local_corr = global_corr

            # CMAQ偏差
            cmaq_dev = np.abs(m_test[i] - obs_mean) / obs_std

            # 结合局部相关性和CMAQ偏差
            # 相关性高 -> 更多信任CMAQ校正
            # 相关性低 -> 更多信任VNA
            corr_factor = (local_corr + global_corr) / 2  # 0到1之间

            # 当相关性高且CMAQ偏离大时，更信任CMAQ校正
            prior_vna = np.clip(0.70 - 0.40 * corr_factor - 0.20 * cmaq_dev, 0.20, 0.70)

            # 似然权重
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

            lcaf_pred[i] = w_vna * vna_i + w_avna * avna_i + w_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'lcaf': lcaf_pred
        }

        print(f"  Fold {fold_id}: completed")

    lcaf_all = np.concatenate([results[f]['lcaf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, lcaf_all)
    print(f"  LocalCorrelationAdaptiveFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_variance_weighted_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    VarianceWeightedFusion: 方差加权融合

    原理:
    - 不仅用残差方差，还考虑局部预测方差
    - 使用更复杂的方差估计
    - 结合全局和局部方差信息
    """
    print("\n=== VarianceWeightedFusion ===")

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

        # 全局残差方差
        vna_pred_train = nn_vna.predict(coords_train)
        avna_pred_train = m_train + nn_bias.predict(coords_train)
        evna_pred_train = m_train * nn_ratio.predict(coords_train)

        residual_vna = y_train - vna_pred_train
        residual_avna = y_train - avna_pred_train
        residual_evna = y_train - evna_pred_train

        global_std_vna = np.std(residual_vna) + 1e-6
        global_std_avna = np.std(residual_avna) + 1e-6
        global_std_evna = np.std(residual_evna) + 1e-6

        # 局部方差估计
        k = 15
        nn_finder = NearestNeighbors(n_neighbors=min(k, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        obs_mean = y_train.mean()
        obs_std = y_train.std()

        vwf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]])

            # 局部残差方差
            local_res_vna = residual_vna[indices[0]]
            local_res_avna = residual_avna[indices[0]]
            local_res_evna = residual_evna[indices[0]]

            local_std_vna = np.std(local_res_vna) + 1e-6
            local_std_avna = np.std(local_res_avna) + 1e-6
            local_std_evna = np.std(local_res_evna) + 1e-6

            # 结合全局和局部方差
            combined_std_vna = 0.5 * global_std_vna + 0.5 * local_std_vna
            combined_std_avna = 0.5 * global_std_avna + 0.5 * local_std_avna
            combined_std_evna = 0.5 * global_std_evna + 0.5 * local_std_evna

            # CMAQ偏差
            cmaq_dev = np.abs(m_test[i] - obs_mean) / obs_std

            # 方差倒数作为权重
            inv_std_vna = 1.0 / combined_std_vna
            inv_std_avna = 1.0 / combined_std_avna
            inv_std_evna = 1.0 / combined_std_evna

            # 基于CMAQ偏差的基础权重
            prior_vna = np.clip(0.68 - 0.32 * cmaq_dev, 0.25, 0.68)

            # 结合方差权重
            total_inv = inv_std_vna + inv_std_avna + inv_std_evna
            var_weight_vna = inv_std_vna / total_inv
            var_weight_avna = inv_std_avna / total_inv
            var_weight_evna = inv_std_evna / total_inv

            # 综合权重
            w_vna = 0.6 * prior_vna + 0.4 * var_weight_vna
            w_avna = (1 - prior_vna) * (0.5 * var_weight_avna / (var_weight_avna + var_weight_evna + 1e-6))
            w_evna = (1 - prior_vna) * (0.5 * var_weight_evna / (var_weight_avna + var_weight_evna + 1e-6))

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

            vwf_pred[i] = w_vna * vna_i + w_avna * avna_i + w_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'vwf': vwf_pred
        }

        print(f"  Fold {fold_id}: completed")

    vwf_all = np.concatenate([results[f]['vwf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, vwf_all)
    print(f"  VarianceWeightedFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_optimized_nonlinear_weighting(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    OptimizedNonlinearWeighting: 优化非线性加权融合

    原理:
    - 使用非线性函数调整权重
    - CMAQ偏差大时更激进地调整权重
    - 使用sigmoid型平滑过渡
    """
    print("\n=== OptimizedNonlinearWeighting ===")

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

        obs_mean = y_train.mean()
        obs_std = y_train.std()

        onw_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            cmaq_dev = np.abs(m_test[i] - obs_mean) / obs_std

            # 非线性权重调整
            # 使用指数衰减型权重
            decay = np.exp(-2.0 * cmaq_dev)

            # sigmoid型平滑过渡
            sigmoid_cmaq = 1.0 / (1.0 + np.exp(-3.0 * (cmaq_dev - 0.5)))

            # 基础prior
            prior_vna = np.clip(0.68 - 0.32 * cmaq_dev, 0.25, 0.68)

            # 非线性调整: 低偏差时保持，高偏差时更信任CMAQ校正
            adjusted_prior_vna = prior_vna * (0.5 + 0.5 * decay)

            # 似然权重
            like_avna = 1.0 / std_avna
            like_evna = 1.0 / std_evna

            w_vna = adjusted_prior_vna
            w_avna = (1 - adjusted_prior_vna) * like_avna / (like_avna + like_evna)
            w_evna = (1 - adjusted_prior_vna) * like_evna / (like_avna + like_evna)

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

            onw_pred[i] = w_vna * vna_i + w_avna * avna_i + w_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'onw': onw_pred
        }

        print(f"  Fold {fold_id}: completed")

    onw_all = np.concatenate([results[f]['onw'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, onw_all)
    print(f"  OptimizedNonlinearWeighting: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("第十四轮：更激进的单模型优化")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    methods = [
        ('EnhancedSpatialStratifiedFusion', run_enhanced_spatial_stratified),
        ('LocalCorrelationAdaptiveFusion', run_local_correlation_adaptive),
        ('VarianceWeightedFusion', run_variance_weighted_fusion),
        ('OptimizedNonlinearWeighting', run_optimized_nonlinear_weighting),
    ]

    for name, func in methods:
        metrics, _ = func(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
        results[name] = metrics

    print("\n" + "="*60)
    print("第十四轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/fourteenth_round_results.csv', index=False)

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