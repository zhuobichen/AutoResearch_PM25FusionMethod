"""
年平均数据 - 第十三轮：新单模型创新方法
==========================================

当前最佳: RefinedBayesianFusion R²=0.5782
目标: R²≥0.5800 (差距0.0018)

新方向:
1. ResidualNormalityCorrection: 残差偏度校正
2. SpatialStratifiedFusion: 空间分层融合
3. QuantileBasedFusion: 分位数加权融合
4. AdaptiveBiasCorrection: 自适应偏差校正

核心约束:
- 单模型（非集成）
- 物理可解释
- R²提升≥0.01
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
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


def run_residual_normality_correction(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    ResidualNormalityCorrection: 残差正态性校正融合

    原理:
    - 分析VNA/aVNA/eVNA残差的偏度和峰度
    - 根据残差分布特征进行偏度/峰度校正
    - 物理意义: 残差分布偏离正态性时进行校正可提升预测
    """
    print("\n=== ResidualNormalityCorrection ===")

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

        # 计算训练集残差
        vna_pred_train = nn_vna.predict(coords_train)
        avna_pred_train = m_train + nn_bias.predict(coords_train)
        evna_pred_train = m_train * nn_ratio.predict(coords_train)

        residual_vna = y_train - vna_pred_train
        residual_avna = y_train - avna_pred_train
        residual_evna = y_train - evna_pred_train

        # 计算偏度和峰度
        skew_vna = stats.skew(residual_vna)
        skew_avna = stats.skew(residual_avna)
        skew_evna = stats.skew(residual_evna)

        kurt_vna = stats.kurtosis(residual_vna)
        kurt_avna = stats.kurtosis(residual_avna)
        kurt_evna = stats.kurtosis(residual_evna)

        obs_mean = y_train.mean()
        obs_std = y_train.std()

        rnc_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            cmaq_dev = np.abs(m_test[i] - obs_mean) / obs_std

            # 偏度校正因子：偏度为正时过估，需下调
            # 偏度为负时欠估，需上调
            corr_vna = 1.0 - 0.1 * skew_vna
            corr_avna = 1.0 - 0.1 * skew_avna
            corr_evna = 1.0 - 0.1 * skew_evna

            # 权重（类似RefinedBayesianFusion）
            prior_vna = np.clip(0.68 - 0.32 * cmaq_dev, 0.28, 0.68)

            like_avna = 1.0 / (np.std(residual_avna) + 1e-6)
            like_evna = 1.0 / (np.std(residual_evna) + 1e-6)

            w_vna = prior_vna
            w_avna = (1 - prior_vna) * like_avna / (like_avna + like_evna + 1e-6)
            w_evna = (1 - prior_vna) * like_evna / (like_avna + like_evna + 1e-6)

            total = w_vna + w_avna + w_evna
            w_vna /= total
            w_avna /= total
            w_evna /= total

            # 提取预测并应用校正
            dist = np.sqrt((lon_cmaq - coords_test[i, 0])**2 + (lat_cmaq - coords_test[i, 1])**2)
            idx = np.argmin(dist)
            row_idx, col_idx = idx // nx, idx % nx
            vna_i = vna_grid[row_idx * nx + col_idx] * corr_vna
            avna_i = avna_grid[row_idx * nx + col_idx] * corr_avna
            evna_i = evna_grid[row_idx * nx + col_idx] * corr_evna

            rnc_pred[i] = w_vna * vna_i + w_avna * avna_i + w_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'rnc': rnc_pred
        }

        print(f"  Fold {fold_id}: completed (skew_vna={skew_vna:.3f}, kurt_vna={kurt_vna:.3f})")

    rnc_all = np.concatenate([results[f]['rnc'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, rnc_all)
    print(f"  ResidualNormalityCorrection: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_spatial_stratified_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    SpatialStratifiedFusion: 空间分层融合

    原理:
    - 根据站点的地理位置和CMAQ偏差特征进行空间分层
    - 不同区域使用不同的融合权重
    - 物理意义: 不同地理区域可能有不同的偏差特征
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

        # 空间分层: 使用CMAQ偏差进行分类
        bias_train = y_train - m_train
        bias_mean = np.mean(bias_train)
        bias_std = np.std(bias_train)

        # 计算每个站点的局部偏差
        k = 15
        nn_finder = NearestNeighbors(n_neighbors=min(k, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        local_bias_classes = np.zeros(len(test_df), dtype=int)
        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]])
            local_bias = np.mean(bias_train[indices[0]])
            # 分类: 0=低偏差区, 1=中偏差区, 2=高偏差区
            if local_bias < bias_mean - 0.5 * bias_std:
                local_bias_classes[i] = 0  # 低偏差
            elif local_bias > bias_mean + 0.5 * bias_std:
                local_bias_classes[i] = 2  # 高偏差
            else:
                local_bias_classes[i] = 1  # 中偏差

        obs_mean = y_train.mean()
        obs_std = y_train.std()

        ssf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            cmaq_dev = np.abs(m_test[i] - obs_mean) / obs_std
            zone = local_bias_classes[i]

            # 不同区域使用不同权重
            if zone == 0:  # 低偏差区 - 更多信任VNA
                prior_vna = np.clip(0.75 - 0.30 * cmaq_dev, 0.35, 0.75)
            elif zone == 2:  # 高偏差区 - 更多信任CMAQ校正
                prior_vna = np.clip(0.55 - 0.35 * cmaq_dev, 0.25, 0.60)
            else:  # 中偏差区 - 中等权重
                prior_vna = np.clip(0.68 - 0.32 * cmaq_dev, 0.28, 0.68)

            # 似然权重
            avna_res = y_train - (m_train + nn_bias.predict(coords_train))
            evna_res = y_train - (m_train * nn_ratio.predict(coords_train))
            like_avna = 1.0 / (np.std(avna_res) + 1e-6)
            like_evna = 1.0 / (np.std(evna_res) + 1e-6)

            w_vna = prior_vna
            w_avna = (1 - prior_vna) * like_avna / (like_avna + like_evna + 1e-6)
            w_evna = (1 - prior_vna) * like_evna / (like_avna + like_evna + 1e-6)

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

            ssf_pred[i] = w_vna * vna_i + w_avna * avna_i + w_evna * evna_i

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


def run_quantile_based_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    QuantileBasedFusion: 分位数加权融合

    原理:
    - 根据CMAQ值的分位数位置调整权重
    - CMAQ极低值区域更多信任VNA
    - CMAQ极高值区域更多信任CMAQ校正
    - 物理意义: 不同污染水平下模型表现不同
    """
    print("\n=== QuantileBasedFusion ===")

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

        # CMAQ分位数
        q25 = np.percentile(m_train, 25)
        q50 = np.percentile(m_train, 50)
        q75 = np.percentile(m_train, 75)

        obs_mean = y_train.mean()
        obs_std = y_train.std()

        qbf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            m_val = m_test[i]
            cmaq_dev = np.abs(m_val - obs_mean) / obs_std

            # 根据CMAQ分位数位置确定基础权重
            if m_val <= q25:  # 低污染区
                prior_vna = np.clip(0.72 - 0.25 * cmaq_dev, 0.35, 0.72)
            elif m_val >= q75:  # 高污染区
                prior_vna = np.clip(0.60 - 0.35 * cmaq_dev, 0.25, 0.62)
            else:  # 中等污染区
                prior_vna = np.clip(0.68 - 0.32 * cmaq_dev, 0.28, 0.68)

            # 似然权重
            avna_res = y_train - (m_train + nn_bias.predict(coords_train))
            evna_res = y_train - (m_train * nn_ratio.predict(coords_train))
            like_avna = 1.0 / (np.std(avna_res) + 1e-6)
            like_evna = 1.0 / (np.std(evna_res) + 1e-6)

            w_vna = prior_vna
            w_avna = (1 - prior_vna) * like_avna / (like_avna + like_evna + 1e-6)
            w_evna = (1 - prior_vna) * like_evna / (like_avna + like_evna + 1e-6)

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

            qbf_pred[i] = w_vna * vna_i + w_avna * avna_i + w_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'qbf': qbf_pred
        }

        print(f"  Fold {fold_id}: completed (q25={q25:.1f}, q50={q50:.1f}, q75={q75:.1f})")

    qbf_all = np.concatenate([results[f]['qbf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, qbf_all)
    print(f"  QuantileBasedFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_adaptive_bias_correction(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    AdaptiveBiasCorrection: 自适应偏差校正融合

    原理:
    - 根据VNA与CMAQ的偏差动态调整校正强度
    - 偏差大时使用更强的CMAQ校正
    - 偏差小时更多信任VNA
    - 物理意义: 局部偏差大时说明VNA在该区域不准确
    """
    print("\n=== AdaptiveBiasCorrection ===")

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

        # 局部VNA-CMAQ偏差
        k = 20
        nn_finder = NearestNeighbors(n_neighbors=min(k, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        obs_mean = y_train.mean()
        obs_std = y_train.std()

        abc_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists, indices = nn_finder.kneighbors([coords_test[i]])

            # 局部VNA预测
            vna_local = nn_vna.predict(coords_train[indices[0]])
            cmaq_local = m_train[indices[0]]

            # 局部偏差强度
            local_bias = np.mean(y_train[indices[0]] - vna_local)
            local_bias_abs = np.abs(local_bias)

            # 根据局部偏差强度调整
            bias_threshold = obs_std * 0.3  # 偏差阈值

            cmaq_dev = np.abs(m_test[i] - obs_mean) / obs_std

            if local_bias_abs < bias_threshold:
                # 局部偏差小 -> 更多信任VNA
                prior_vna = np.clip(0.75 - 0.30 * cmaq_dev, 0.40, 0.75)
            else:
                # 局部偏差大 -> 更多信任CMAQ校正
                prior_vna = np.clip(0.58 - 0.28 * cmaq_dev, 0.22, 0.58)

            # 似然权重
            avna_res = y_train - (m_train + nn_bias.predict(coords_train))
            evna_res = y_train - (m_train * nn_ratio.predict(coords_train))
            like_avna = 1.0 / (np.std(avna_res) + 1e-6)
            like_evna = 1.0 / (np.std(evna_res) + 1e-6)

            w_vna = prior_vna
            w_avna = (1 - prior_vna) * like_avna / (like_avna + like_evna + 1e-6)
            w_evna = (1 - prior_vna) * like_evna / (like_avna + like_evna + 1e-6)

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

            abc_pred[i] = w_vna * vna_i + w_avna * avna_i + w_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'abc': abc_pred
        }

        print(f"  Fold {fold_id}: completed")

    abc_all = np.concatenate([results[f]['abc'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, abc_all)
    print(f"  AdaptiveBiasCorrection: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("第十三轮：新单模型创新方法")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    methods = [
        ('ResidualNormalityCorrection', run_residual_normality_correction),
        ('SpatialStratifiedFusion', run_spatial_stratified_fusion),
        ('QuantileBasedFusion', run_quantile_based_fusion),
        ('AdaptiveBiasCorrection', run_adaptive_bias_correction),
    ]

    for name, func in methods:
        metrics, _ = func(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
        results[name] = metrics

    print("\n" + "="*60)
    print("第十三轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/thirteenth_round_results.csv', index=False)

    print("\n方法排名 (按R2降序):")
    for _, row in results_df.iterrows():
        print(f"  {row['method']:30s}: R2={row['R2']:.4f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}, MB={row['MB']:.2f}")

    # 基准对比
    print("\n" + "="*60)
    print("基准对比 (VNA R2=0.5700)")
    print("="*60)
    for _, row in results_df.iterrows():
        improvement = row['R2'] - 0.5700
        status = "达标" if improvement >= 0.01 else "未达标"
        print(f"  {row['method']:30s}: 提升 {improvement:+.4f} [{status}]")

    # 与最佳对比
    print("\n" + "="*60)
    print("与最佳单模型对比 (RefinedBayesianFusion R2=0.5782)")
    print("="*60)
    for _, row in results_df.iterrows():
        diff = row['R2'] - 0.5782
        print(f"  {row['method']:30s}: 差距 {diff:+.4f}")

    return results_df


if __name__ == '__main__':
    results_df = main()