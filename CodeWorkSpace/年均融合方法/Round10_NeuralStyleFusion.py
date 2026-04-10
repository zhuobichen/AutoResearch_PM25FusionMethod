"""
年平均数据 - 第十轮：核感知融合与增强残差学习
==========================================

当前最佳 (排除集成方法):
- CMAQConditionedFusion: R²=0.5759
- LocalCorrelationFusion: R²=0.5744
- SpatialLocalWeighting: R²=0.5744

核心理念:
V6-Ensemble有效的原因: 根据CMAQ偏差动态调整VNA和aVNA权重
问题: 如何用单一模型捕捉这种动态调整?

新方法:
1. KernelAwareFusion: 核感知融合 - 学习空间变化的关系
2. ResidualBoostingFusion: 残差提升融合 - 迭代细化残差
3. SignalDecompositionFusion: 信号分解融合 - 分离空间尺度的信号
4. DistanceAwareGPR: 距离感知GPR - 根据距离调整CMAQ权重
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
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


def run_kernel_aware_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    KernelAwareFusion: 核感知融合
    思想: 使用CMAQ感知的高斯核来加权邻居，而非纯空间距离
    """
    print("\n=== KernelAwareFusion ===")

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

        # eVNA: M * r, 其中r是空间插值的比率
        ratio = y_train / m_train
        nn_ratio = NNA(method='nearest', k=15, power=-2)
        nn_ratio.fit(coords_train, ratio)
        ratio_grid = nn_ratio.predict(X_grid_full)
        evna_grid = y_grid_model_full * ratio_grid

        # aVNA: M + b, 其中b是空间插值的偏差
        bias = y_train - m_train
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(coords_train, bias)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_grid = y_grid_model_full + bias_grid

        # VNA: 纯空间插值
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(coords_train, y_train)
        vna_grid = nn_vna.predict(X_grid_full)

        # 核心: 根据CMAQ偏差确定加权核
        obs_mean = y_train.mean()
        obs_std = y_train.std()

        kaf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            # CMAQ偏差
            cmaq_dev = (m_test[i] - obs_mean) / obs_std

            # 根据CMAQ偏差动态调整融合权重
            # 偏差大 -> 更信任aVNA (CMAQ偏离正常值)
            # 偏差小 -> 更信任VNA (空间插值更可靠)
            w_vna = np.clip(0.65 - 0.3 * np.abs(cmaq_dev), 0.25, 0.65)
            w_avna = np.clip(0.2 + 0.3 * np.abs(cmaq_dev), 0.2, 0.5)
            w_evna = 1 - w_vna - w_avna

            # 提取网格预测
            dist = np.sqrt((lon_cmaq - coords_test[i, 0])**2 + (lat_cmaq - coords_test[i, 1])**2)
            idx = np.argmin(dist)
            row_idx, col_idx = idx // nx, idx % nx
            vna_i = vna_grid[row_idx * nx + col_idx]
            avna_i = avna_grid[row_idx * nx + col_idx]
            evna_i = evna_grid[row_idx * nx + col_idx]

            kaf_pred[i] = w_vna * vna_i + w_avna * avna_i + w_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'kaf': kaf_pred
        }

        print(f"  Fold {fold_id}: completed")

    kaf_all = np.concatenate([results[f]['kaf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, kaf_all)
    print(f"  KernelAwareFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_residual_boosting_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    ResidualBoostingFusion: 残差提升融合
    思想: 迭代细化残差，每次关注空间上一个尺度的结构
    """
    print("\n=== ResidualBoostingFusion ===")

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

        # 第一阶段: 全局趋势 O = a + b*M
        lr = LinearRegression()
        lr.fit(m_train.reshape(-1, 1), y_train)
        trend_pred = lr.intercept_ + lr.coef_[0] * m_test
        residual = y_train - lr.predict(m_train.reshape(-1, 1))

        # 第二阶段: 局部校正 - k=15
        nn15 = NNA(method='nearest', k=15, power=-2)
        nn15.fit(coords_train, residual)
        residual_pred_15 = nn15.predict(coords_test)
        pred_15 = trend_pred + residual_pred_15

        # 第三阶段: 更局部的校正 - k=8
        residual_2 = y_train - (lr.predict(m_train.reshape(-1, 1)) + nn15.predict(coords_train))
        nn8 = NNA(method='nearest', k=8, power=-2)
        nn8.fit(coords_train, residual_2)
        residual_pred_8 = nn8.predict(coords_test)

        rbf_pred = pred_15 + 0.5 * residual_pred_8

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'rbf': rbf_pred
        }

        print(f"  Fold {fold_id}: completed")

    rbf_all = np.concatenate([results[f]['rbf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, rbf_all)
    print(f"  ResidualBoostingFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_signal_decomposition_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    SignalDecompositionFusion: 信号分解融合
    思想: 将观测分解为趋势(CMAQ) + 空间结构(残差), 分别建模后融合
    """
    print("\n=== SignalDecompositionFusion ===")

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

        # 1. 全局趋势
        lr = LinearRegression()
        lr.fit(np.column_stack([m_train, m_train**2]), y_train)
        trend_train = lr.predict(np.column_stack([m_train, m_train**2]))
        trend_test = lr.predict(np.column_stack([m_test, m_test**2]))

        # 2. 空间残差1: 大尺度 (k=30)
        residual_1 = y_train - trend_train
        nn_large = NNA(method='nearest', k=30, power=-2)
        nn_large.fit(coords_train, residual_1)
        residual_1_test = nn_large.predict(coords_test)

        # 3. 空间残差2: 中尺度 (k=15)
        residual_2 = residual_1 - nn_large.predict(coords_train)
        nn_medium = NNA(method='nearest', k=15, power=-2)
        nn_medium.fit(coords_train, residual_2)
        residual_2_test = nn_medium.predict(coords_test)

        # 4. 空间残差3: 小尺度 (k=5)
        residual_3 = residual_2 - nn_medium.predict(coords_train)
        nn_small = NNA(method='nearest', k=5, power=-2)
        nn_small.fit(coords_train, residual_3)
        residual_3_test = nn_small.predict(coords_test)

        # 融合: 趋势 + 加权残差
        sdf_pred = trend_test + 0.8 * residual_1_test + 0.5 * residual_2_test + 0.3 * residual_3_test

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'sdf': sdf_pred
        }

        print(f"  Fold {fold_id}: completed")

    sdf_all = np.concatenate([results[f]['sdf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, sdf_all)
    print(f"  SignalDecompositionFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_distance_aware_gpr(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    DistanceAwareGPR: 距离感知GPR
    思想: 在GPR中, 根据到训练点的距离调整CMAQ的权重
    """
    print("\n=== DistanceAwareGPR ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    # GPR kernel
    kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

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

        # 第一步: 计算CMAQ偏差 (O-M)
        bias = y_train - m_train

        # 第二步: GPR建模偏差
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1.0, normalize_y=True)
        gpr.fit(coords_train, bias)
        bias_pred, _ = gpr.predict(coords_test, return_std=True)

        # 第三步: 根据预测不确定性调整
        # GPR的std代表空间不确定性
        std_pred, _ = gpr.predict(coords_test, return_std=True)

        # 基础预测: M + bias
        dag_pred_base = m_test + bias_pred

        # VNA预测
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(coords_train, y_train)
        vna_pred = nn_vna.predict(coords_test)

        # 根据GPR不确定性混合
        # 不确定性高 -> 更多依赖VNA
        # 不确定性低 -> 更多依赖M+bias
        max_std = std_pred.max() + 1e-6
        uncertainty = std_pred / max_std
        weight_bias = 1 - uncertainty

        dag_pred = weight_bias * dag_pred_base + uncertainty * vna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'dag': dag_pred
        }

        print(f"  Fold {fold_id}: completed")

    dag_all = np.concatenate([results[f]['dag'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, dag_all)
    print(f"  DistanceAwareGPR: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_cmaq_ratio_smooth_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    CMAQRatioSmoothFusion: CMAQ比率平滑融合
    思想: eVNA的比率场在空间上应该是平滑的，用平滑后的比率预测
    """
    print("\n=== CMAQRatioSmoothFusion ===")

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

        # 计算比率
        ratio = y_train / m_train

        # 多尺度平滑比率
        scales = [10, 20, 40]
        weights = [0.3, 0.4, 0.3]

        ratio_grid_sum = np.zeros(len(y_grid_model_full))
        for k, w in zip(scales, weights):
            nn = NNA(method='nearest', k=k, power=-2)
            nn.fit(coords_train, ratio)
            ratio_grid = nn.predict(X_grid_full)
            ratio_grid_sum += w * ratio_grid

        # eVNA预测
        evna_grid = y_grid_model_full * ratio_grid_sum
        evna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, evna_grid, ny, nx)

        # aVNA预测
        bias = y_train - m_train
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(coords_train, bias)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_grid = y_grid_model_full + bias_grid
        avna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, avna_grid, ny, nx)

        # VNA预测
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(coords_train, y_train)
        vna_pred = nn_vna.predict(coords_test)

        # CMAQ引导的动态权重
        obs_mean = y_train.mean()
        obs_std = y_train.std()
        cmaq_dev = np.abs(m_test - obs_mean) / obs_std

        # 平滑的权重转换
        w_vna = np.clip(0.6 - 0.3 * cmaq_dev, 0.25, 0.6)
        w_avna = np.clip(0.2 + 0.25 * cmaq_dev, 0.2, 0.45)
        w_evna = 1 - w_vna - w_avna

        crsf_pred = w_vna * vna_pred + w_avna * avna_pred + w_evna * evna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'crsf': crsf_pred
        }

        print(f"  Fold {fold_id}: completed")

    crsf_all = np.concatenate([results[f]['crsf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, crsf_all)
    print(f"  CMAQRatioSmoothFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("第十轮：核感知融合与增强残差学习")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    methods = [
        ('KernelAwareFusion', run_kernel_aware_fusion),
        ('ResidualBoostingFusion', run_residual_boosting_fusion),
        ('SignalDecompositionFusion', run_signal_decomposition_fusion),
        ('DistanceAwareGPR', run_distance_aware_gpr),
        ('CMAQRatioSmoothFusion', run_cmaq_ratio_smooth_fusion),
    ]

    for name, func in methods:
        metrics, _ = func(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
        results[name] = metrics

    print("\n" + "="*60)
    print("第十轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/tenth_round_results.csv', index=False)

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