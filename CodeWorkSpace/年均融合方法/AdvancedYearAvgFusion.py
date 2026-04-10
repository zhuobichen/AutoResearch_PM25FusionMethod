"""
年平均数据高级融合方法 (第二轮)
===============================
基于上一轮分析，开发更有效的创新方法

核心洞察:
1. VNA (纯空间插值) R2=0.5700 是最强的基准
2. 年平均数据的CMAQ偏差可能更稳定，需要更简单的校正
3. 关键是如何有效融合空间插值和CMAQ模型

新方法:
1. BiasCorrectedVNA: VNA + 简单偏差校正
2. RatioCorrectedVNA: VNA + 比率校正
3. CMAQGuidedFusion: CMAQ引导的空间融合
4. LocalBiasFusion: 局部偏差融合
5. VarianceWeightedFusion: 方差加权融合
6. EnsembleVNA: VNA集合
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
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
    print("=== Loading and Preparing Year-Average Data ===")

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

    return monitor_year_avg, lon_cmaq, lat_cmaq, cmaq_year_avg


def run_bias_corrected_vna(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    BiasCorrectedVNA: VNA + 简单线性偏差校正
    核心思想：先用VNA捕获空间结构，再用线性模型校正整体偏差
    """
    print("\n=== Bias Corrected VNA ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # Step 1: VNA空间插值
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(train_df[['Lon', 'Lat']].values, train_df['Conc'].values)
        vna_pred_grid = nn_vna.predict(X_grid_full)

        # 提取VNA预测
        vna_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            vna_pred[i] = vna_pred_grid[idx // nx * nx + idx % nx]

        # Step 2: 计算残差并建模
        # residual = O - VNA_pred
        residual_train = train_df['Conc'].values - nn_vna.predict(train_df[['Lon', 'Lat']].values)

        # 残差与CMAQ偏差相关
        cmaq_deviation = train_df['CMAQ'].values - train_df['Conc'].values

        # 线性回归：residual = a + b * cmaq_deviation
        lr = LinearRegression()
        X_train_res = np.column_stack([np.ones(len(train_df)), cmaq_deviation])
        lr.fit(X_train_res, residual_train)

        # 预测测试集的残差校正
        cmaq_dev_test = test_df['CMAQ'].values - vna_pred
        X_test_res = np.column_stack([np.ones(len(test_df)), cmaq_dev_test])
        residual_correction = lr.predict(X_test_res)

        bcvna_pred = vna_pred + residual_correction

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'bcvna': bcvna_pred
        }

        print(f"  Fold {fold_id}: completed")

    bcvna_all = np.concatenate([results[f]['bcvna'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, bcvna_all)
    print(f"  Bias Corrected VNA: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_ratio_corrected_vna(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    RatioCorrectedVNA: VNA + 比率校正
    """
    print("\n=== Ratio Corrected VNA ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # Step 1: VNA空间插值
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(train_df[['Lon', 'Lat']].values, train_df['Conc'].values)
        vna_pred_grid = nn_vna.predict(X_grid_full)

        # 提取VNA预测
        vna_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            vna_pred[i] = vna_pred_grid[idx // nx * nx + idx % nx]

        # Step 2: 计算log比率残差
        # log_ratio = log(O / VNA_pred)
        vna_train = nn_vna.predict(train_df[['Lon', 'Lat']].values)
        log_ratio_train = np.log(train_df['Conc'].values / np.maximum(vna_train, 1e-6))

        # 比率与CMAQ比率相关
        cmaq_ratio = train_df['CMAQ'].values / np.maximum(vna_train, 1e-6)

        # 线性回归：log_ratio = a + b * log(cmaq_ratio)
        log_cmaq_ratio = np.log(cmaq_ratio)
        lr = LinearRegression()
        X_train_res = np.column_stack([np.ones(len(train_df)), log_cmaq_ratio])
        lr.fit(X_train_res, log_ratio_train)

        # 预测测试集的比率校正
        cmaq_ratio_test = test_df['CMAQ'].values / np.maximum(vna_pred, 1e-6)
        log_cmaq_ratio_test = np.log(cmaq_ratio_test)
        X_test_res = np.column_stack([np.ones(len(test_df)), log_cmaq_ratio_test])
        log_ratio_correction = lr.predict(X_test_res)

        rcvna_pred = vna_pred * np.exp(log_ratio_correction)

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'rcvna': rcvna_pred
        }

        print(f"  Fold {fold_id}: completed")

    rcvna_all = np.concatenate([results[f]['rcvna'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, rcvna_all)
    print(f"  Ratio Corrected VNA: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_cmaq_guided_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    CMAQGuidedFusion: CMAQ引导的空间融合
    思想：当CMAQ值接近观测均值时，更信任VNA；当CMAQ偏离时，用CMAQ校正
    """
    print("\n=== CMAQ Guided Fusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # VNA预测
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(train_df[['Lon', 'Lat']].values, train_df['Conc'].values)
        vna_pred_grid = nn_vna.predict(X_grid_full)

        # aVNA预测 (CMAQ + bias)
        train_df = train_df.copy()
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(train_df[['Lon', 'Lat']].values, train_df['bias'].values)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_pred_grid = y_grid_model_full + bias_grid

        # 提取测试站点预测
        vna_pred = np.zeros(len(test_df))
        avna_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            vna_pred[i] = vna_pred_grid[idx // nx * nx + idx % nx]
            avna_pred[i] = avna_pred_grid[idx // nx * nx + idx % nx]

        # CMAQ引导的动态权重
        # 当CMAQ接近观测均值时，信任VNA；偏离时，信任aVNA
        obs_mean = train_df['Conc'].mean()
        cmaq_test = test_df['CMAQ'].values

        # 计算偏差程度
        deviation = np.abs(cmaq_test - obs_mean) / obs_mean
        # 动态权重：偏差小用VNA，偏差大用aVNA
        weight_vna = np.clip(1 - deviation, 0.2, 0.8)

        cgf_pred = weight_vna * vna_pred + (1 - weight_vna) * avna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'cgf': cgf_pred
        }

        print(f"  Fold {fold_id}: completed")

    cgf_all = np.concatenate([results[f]['cgf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, cgf_all)
    print(f"  CMAQ Guided Fusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_local_bias_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    LocalBiasFusion: 局部偏差融合
    思想：计算每个站点的局部偏差，然后空间插值这个偏差到网格
    """
    print("\n=== Local Bias Fusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # Step 1: 计算每个训练站点的局部偏差
        # 局部偏差 = O - M (观测减模型)
        train_df['local_bias'] = train_df['Conc'] - train_df['CMAQ']

        # Step 2: 空间插值局部偏差到网格
        nn_bias = NNA(method='nearest', k=25, power=-2)  # 增加k以获取更平滑的局部偏差
        nn_bias.fit(train_df[['Lon', 'Lat']].values, train_df['local_bias'].values)
        local_bias_grid = nn_bias.predict(X_grid_full)

        # Step 3: 融合预测 = CMAQ网格 + 局部偏差网格
        lbf_pred_grid = y_grid_model_full + local_bias_grid

        # Step 4: 提取测试站点预测
        lbf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            lbf_pred[i] = lbf_pred_grid[idx // nx * nx + idx % nx]

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'lbf': lbf_pred
        }

        print(f"  Fold {fold_id}: completed")

    lbf_all = np.concatenate([results[f]['lbf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, lbf_all)
    print(f"  Local Bias Fusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_variance_weighted_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    VarianceWeightedFusion: 基于局部方差加权融合
    思想：在CMAQ方差大的区域，更信任VNA；在CMAQ方差小的区域，更信任CMAQ校正
    """
    print("\n=== Variance Weighted Fusion ===")

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

        # 计算CMAQ网格的局部方差
        # 用CMAQ的局部梯度作为方差的代理
        cmaq_gradient = np.gradient(cmaq_year_avg, axis=0) ** 2 + np.gradient(cmaq_year_avg, axis=1) ** 2
        cmaq_variance_grid = cmaq_gradient.ravel()

        # 归一化方差权重
        variance_weight = cmaq_variance_grid / (cmaq_variance_grid.max() + 1e-6)
        # 方差大时更信任VNA (空间插值)，方差小时更信任aVNA
        vna_weight = np.clip(0.5 + 0.3 * variance_weight, 0.3, 0.7)

        # 融合
        fusion_grid = vna_weight * vna_pred_grid + (1 - vna_weight) * avna_pred_grid

        # 提取测试站点预测
        vwf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            vwf_pred[i] = fusion_grid[idx // nx * nx + idx % nx]

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'vwf': vwf_pred
        }

        print(f"  Fold {fold_id}: completed")

    vwf_all = np.concatenate([results[f]['vwf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, vwf_all)
    print(f"  Variance Weighted Fusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_ensemble_vna(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    EnsembleVNA: VNA集合，使用不同的k值和power参数
    """
    print("\n=== Ensemble VNA ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 多个VNA配置
        configs = [
            {'k': 10, 'power': -2},
            {'k': 15, 'power': -2},
            {'k': 20, 'power': -2},
            {'k': 15, 'power': -1.5},
            {'k': 15, 'power': -2.5},
        ]

        ensemble_preds = []
        for cfg in configs:
            nn = NNA(method='nearest', k=cfg['k'], power=cfg['power'])
            nn.fit(train_df[['Lon', 'Lat']].values, train_df['Conc'].values)
            pred_grid = nn.predict(X_grid_full)

            pred = np.zeros(len(test_df))
            for i in range(len(test_df)):
                dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
                idx = np.argmin(dist)
                pred[i] = pred_grid[idx // nx * nx + idx % nx]
            ensemble_preds.append(pred)

        # 平均
        evna_pred = np.mean(ensemble_preds, axis=0)

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'evna': evna_pred
        }

        print(f"  Fold {fold_id}: completed")

    evna_all = np.concatenate([results[f]['evna'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, evna_all)
    print(f"  Ensemble VNA: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("年平均数据高级融合方法测试 (第二轮)")
    print("="*60)

    # 加载数据
    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    # 运行各种方法
    methods = [
        ('BiasCorrectedVNA', run_bias_corrected_vna),
        ('RatioCorrectedVNA', run_ratio_corrected_vna),
        ('CMAQGuidedFusion', run_cmaq_guided_fusion),
        ('LocalBiasFusion', run_local_bias_fusion),
        ('VarianceWeightedFusion', run_variance_weighted_fusion),
        ('EnsembleVNA', run_ensemble_vna),
    ]

    for name, func in methods:
        metrics, _ = func(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
        results[name] = metrics

    # 保存结果
    print("\n" + "="*60)
    print("第二轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/advanced_results.csv', index=False)

    print("\n方法排名 (按R2降序):")
    for _, row in results_df.iterrows():
        print(f"  {row['method']:25s}: R2={row['R2']:.4f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}")

    return results_df


if __name__ == '__main__':
    results_df = main()