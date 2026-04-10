"""
年平均数据 - 第八轮：真正的单模型创新
=====================================

已验证方法的局限性:
- GPR直接建模: R²=0.5623 (CMAQ作为特征效果不佳)
- Matern GPR: R²=0.3498 (核函数选择不当)
- 分位数回归: R²=0.4057 (对偏态数据处理有问题)

真正创新的单模型方向:
1. PolynomialGPRFusion: CMAQ多项式特征 + GPR空间残差
2. VariogramKrigingFusion: 变差函数建模的空间结构
3. CMAQConditionedFusion: CMAQ条件化的空间建模
4. HybridTrendResidual: 混合趋势+残差的空间插值
5. SpatialLocalWeighting: 空间局部加权融合
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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


def run_polynomial_gpr_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    PolynomialGPRFusion: 多项式趋势 + GPR空间残差
    核心思想: 使用CMAQ的多项式来捕捉非线性偏差，然后用GPR建模空间残差
    O = a + b*M + c*M^2 + GPR(Lon, Lat)
    """
    print("\n=== PolynomialGPRFusion ===")

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

        # 多项式趋势: M, M^2
        m_train = train_df['CMAQ'].values
        m_test = test_df['CMAQ'].values

        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        # 线性回归
        lr = LinearRegression()
        lr.fit(m_train_poly, train_df['Conc'].values)
        trend_pred = lr.predict(m_test_poly)
        trend_residual = train_df['Conc'].values - lr.predict(m_train_poly)

        # GPR建模残差 (仅用空间坐标)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1.0, normalize_y=True)
        gpr.fit(train_df[['Lon', 'Lat']].values, trend_residual)
        gpr_pred, _ = gpr.predict(test_df[['Lon', 'Lat']].values, return_std=True)

        pgf_pred = trend_pred + gpr_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'pgf': pgf_pred
        }

        print(f"  Fold {fold_id}: completed")

    pgf_all = np.concatenate([results[f]['pgf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, pgf_all)
    print(f"  PolynomialGPRFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_variogram_kriging_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    VariogramKrigingFusion: 变差函数引导的空间插值
    核心思想: 使用各向异性变差函数来建模空间相关性
    """
    print("\n=== VariogramKrigingFusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 计算实验变差函数
        coords_train = train_df[['Lon', 'Lat']].values
        values_train = train_df['Conc'].values

        # 计算残差 (O - M)
        residual = values_train - train_df['CMAQ'].values

        # 使用多个方向的IDW来模拟变差函数效应
        # 方向性权重: N-S, E-W, NE-SW, NW-SE
        def directional_weights(test_coord, train_coords, direction='NS'):
            """计算方向性权重"""
            if direction == 'NS':
                delta = np.abs(train_coords[:, 1] - test_coord[1])
            elif direction == 'EW':
                delta = np.abs(train_coords[:, 0] - test_coord[0])
            elif direction == 'NE':
                delta = np.abs((train_coords[:, 0] - test_coord[0]) + (train_coords[:, 1] - test_coord[1])) / np.sqrt(2)
            else:  # NW
                delta = np.abs((train_coords[:, 0] - test_coord[0]) - (train_coords[:, 1] - test_coord[1])) / np.sqrt(2)

            weights = 1 / (delta ** 2 + 1e-6)
            return weights / weights.sum()

        # 预测残差
        residual_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            test_coord = test_df.iloc[i][['Lon', 'Lat']].values

            # 综合四个方向的预测
            preds = []
            for direction in ['NS', 'EW', 'NE', 'NW']:
                weights = directional_weights(test_coord, coords_train, direction)
                preds.append(np.sum(weights * residual))

            residual_pred[i] = np.mean(preds)

        # CMAQ趋势
        vkf_pred = test_df['CMAQ'].values + residual_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'vkf': vkf_pred
        }

        print(f"  Fold {fold_id}: completed")

    vkf_all = np.concatenate([results[f]['vkf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, vkf_all)
    print(f"  VariogramKrigingFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_cmaq_conditioned_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    CMAQConditionedFusion: CMAQ条件化的空间融合
    核心思想: 根据CMAQ值的区间，选择不同的空间插值策略
    """
    print("\n=== CMAQConditionedFusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # VNA
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(train_df[['Lon', 'Lat']].values, train_df['Conc'].values)
        vna_pred_grid = nn_vna.predict(X_grid_full)

        # aVNA
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(train_df[['Lon', 'Lat']].values, train_df['bias'].values)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_pred_grid = y_grid_model_full + bias_grid

        # 提取预测
        vna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, vna_pred_grid, ny, nx)
        avna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, avna_pred_grid, ny, nx)

        # 根据CMAQ分位数确定融合策略
        cmaq_low = train_df['CMAQ'].quantile(0.33)
        cmaq_high = train_df['CMAQ'].quantile(0.67)

        cmaq_test = test_df['CMAQ'].values
        obs_mean = train_df['Conc'].mean()

        # 当CMAQ值偏低时，用aVNA校正比例
        # 当CMAQ值偏高时，用VNA插值
        # 中间区域用混合
        weight_vna = np.zeros(len(test_df))
        for i in range(len(test_df)):
            if cmaq_test[i] < cmaq_low:
                # 低CMAQ区域: 更多依赖aVNA的比率校正
                weight_vna[i] = 0.3
            elif cmaq_test[i] > cmaq_high:
                # 高CMAQ区域: 更多依赖VNA空间插值
                weight_vna[i] = 0.7
            else:
                # 中间区域: 线性过渡
                t = (cmaq_test[i] - cmaq_low) / (cmaq_high - cmaq_low)
                weight_vna[i] = 0.3 + 0.4 * t

        ccf_pred = weight_vna * vna_pred + (1 - weight_vna) * avna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'ccf': ccf_pred
        }

        print(f"  Fold {fold_id}: completed")

    ccf_all = np.concatenate([results[f]['ccf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, ccf_all)
    print(f"  CMAQConditionedFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_hybrid_trend_residual(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    HybridTrendResidual: 混合趋势面+残差的创新融合
    核心思想: 分离全局趋势(CMAQ+空间)和局部残差，分别插值后融合
    """
    print("\n=== HybridTrendResidual ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 全局趋势: CMAQ + Lat + Lon + Lat*Lon
        X_train = np.column_stack([
            train_df['CMAQ'].values,
            train_df['Lat'].values,
            train_df['Lon'].values,
            train_df['Lat'].values * train_df['Lon'].values
        ])

        lr = LinearRegression()
        lr.fit(X_train, train_df['Conc'].values)
        trend_train = lr.predict(X_train)
        residual_train = train_df['Conc'].values - trend_train

        # 空间插值残差
        nn_res = NNA(method='nearest', k=20, power=-2.5)  # 稍大的power使残差更局部
        nn_res.fit(train_df[['Lon', 'Lat']].values, residual_train)
        residual_grid = nn_res.predict(X_grid_full)

        # 网格趋势预测
        X_grid = np.column_stack([
            y_grid_model_full,
            lat_cmaq.ravel(),
            lon_cmaq.ravel(),
            lat_cmaq.ravel() * lon_cmaq.ravel()
        ])
        trend_grid = lr.predict(X_grid)

        # 融合: 趋势 + 残差
        fusion_grid = trend_grid + residual_grid

        htr_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, fusion_grid, ny, nx)

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'htr': htr_pred
        }

        print(f"  Fold {fold_id}: completed")

    htr_all = np.concatenate([results[f]['htr'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, htr_all)
    print(f"  HybridTrendResidual: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_spatial_local_weighting(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    SpatialLocalWeighting: 空间局部加权融合
    核心思想: 根据每个站点的空间邻域特征，动态调整融合权重
    """
    print("\n=== SpatialLocalWeighting ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # VNA
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(train_df[['Lon', 'Lat']].values, train_df['Conc'].values)
        vna_pred_grid = nn_vna.predict(X_grid_full)

        # aVNA
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(train_df[['Lon', 'Lat']].values, train_df['bias'].values)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_pred_grid = y_grid_model_full + bias_grid

        # 提取网格预测
        vna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, vna_pred_grid, ny, nx)
        avna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, avna_pred_grid, ny, nx)

        # 计算每个测试站点的空间邻域密度
        # 使用k近邻的平均距离作为密度指标
        from sklearn.neighbors import NearestNeighbors

        coords_train = train_df[['Lon', 'Lat']].values
        coords_test = test_df[['Lon', 'Lat']].values

        nn_finder = NearestNeighbors(n_neighbors=min(10, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)
        distances, _ = nn_finder.kneighbors(coords_test)
        avg_distances = distances.mean(axis=1)

        # 邻域距离大 -> 空间插值置信度低 -> 更多依赖aVNA
        # 邻域距离小 -> 空间插值置信度高 -> 更多依赖VNA
        max_dist = avg_distances.max() + 1e-6
        confidence = 1 - (avg_distances / max_dist)  # [0, 1]范围

        # 权重: VNA权重随置信度增加
        weight_vna = 0.3 + 0.4 * confidence

        slw_pred = weight_vna * vna_pred + (1 - weight_vna) * avna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'slw': slw_pred
        }

        print(f"  Fold {fold_id}: completed")

    slw_all = np.concatenate([results[f]['slw'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, slw_all)
    print(f"  SpatialLocalWeighting: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("第八轮：真正的单模型创新")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    methods = [
        ('PolynomialGPRFusion', run_polynomial_gpr_fusion),
        ('VariogramKrigingFusion', run_variogram_kriging_fusion),
        ('CMAQConditionedFusion', run_cmaq_conditioned_fusion),
        ('HybridTrendResidual', run_hybrid_trend_residual),
        ('SpatialLocalWeighting', run_spatial_local_weighting),
    ]

    for name, func in methods:
        metrics, _ = func(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
        results[name] = metrics

    print("\n" + "="*60)
    print("第八轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/eighth_round_results.csv', index=False)

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