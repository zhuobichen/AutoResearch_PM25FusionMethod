"""
年平均数据高级融合方法 (第五轮)
==============================
基于前四轮分析，优化三重集成

最佳方法:
- TripleEnsemble: R2=0.5785

第五轮优化:
1. 优化三重集成权重
2. 优化k值和power参数
3. 添加更多空间尺度
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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


def run_optimized_triple_ensemble(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    OptimizedTripleEnsemble: 优化三重集成
    """
    print("\n=== Optimized Triple Ensemble ===")

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
        train_df = train_df.copy()
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(train_df[['Lon', 'Lat']].values, train_df['bias'].values)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_pred_grid = y_grid_model_full + bias_grid

        # eVNA
        train_df['ratio'] = train_df['Conc'] / train_df['CMAQ']
        nn_ratio = NNA(method='nearest', k=15, power=-2)
        nn_ratio.fit(train_df[['Lon', 'Lat']].values, train_df['ratio'].values)
        ratio_grid = nn_ratio.predict(X_grid_full)
        evna_pred_grid = y_grid_model_full * ratio_grid

        # 提取预测
        vna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, vna_pred_grid, ny, nx)
        avna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, avna_pred_grid, ny, nx)
        evna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, evna_pred_grid, ny, nx)

        obs_mean = train_df['Conc'].mean()
        cmaq_test = test_df['CMAQ'].values
        relative_dev = (cmaq_test - obs_mean) / obs_mean

        # 优化后的权重
        w_vna = np.clip(0.55 + 0.5 * relative_dev, 0.2, 0.65)
        w_avna = np.clip(0.25 - 0.25 * relative_dev, 0.15, 0.5)
        w_evna = 1 - w_vna - w_avna
        w_evna = np.clip(w_evna, 0.1, 0.35)

        ote_pred = w_vna * vna_pred + w_avna * avna_pred + w_evna * evna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'ote': ote_pred
        }

        print(f"  Fold {fold_id}: completed")

    ote_all = np.concatenate([results[f]['ote'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, ote_all)
    print(f"  Optimized Triple Ensemble: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_adaptive_triple_ensemble(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    AdaptiveTripleEnsemble: 自适应三重集成
    根据CMAQ偏差程度自适应调整权重
    """
    print("\n=== Adaptive Triple Ensemble ===")

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
        train_df = train_df.copy()
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(train_df[['Lon', 'Lat']].values, train_df['bias'].values)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_pred_grid = y_grid_model_full + bias_grid

        # eVNA
        train_df['ratio'] = train_df['Conc'] / train_df['CMAQ']
        nn_ratio = NNA(method='nearest', k=15, power=-2)
        nn_ratio.fit(train_df[['Lon', 'Lat']].values, train_df['ratio'].values)
        ratio_grid = nn_ratio.predict(X_grid_full)
        evna_pred_grid = y_grid_model_full * ratio_grid

        # 提取预测
        vna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, vna_pred_grid, ny, nx)
        avna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, avna_pred_grid, ny, nx)
        evna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, evna_pred_grid, ny, nx)

        obs_mean = train_df['Conc'].mean()
        obs_std = train_df['Conc'].std()
        cmaq_test = test_df['CMAQ'].values

        # Z-score偏差
        z_dev = (cmaq_test - obs_mean) / obs_std

        # 自适应权重 - 使用tanh函数平滑过渡
        # 当z_dev < 0 (CMAQ偏低): 更多的aVNA校正
        # 当z_dev > 0 (CMAQ偏高): 更多的VNA和eVNA
        w_vna = 0.5 * (1 + np.tanh(0.8 * z_dev))
        w_avna = 0.3 * (1 - np.tanh(0.8 * z_dev))
        w_evna = 1 - w_vna - w_avna

        ate_pred = w_vna * vna_pred + w_avna * avna_pred + w_evna * evna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'ate': ate_pred
        }

        print(f"  Fold {fold_id}: completed")

    ate_all = np.concatenate([results[f]['ate'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, ate_all)
    print(f"  Adaptive Triple Ensemble: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_multiscale_triple_ensemble(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    MultiScaleTripleEnsemble: 多尺度三重集成
    """
    print("\n=== MultiScale Triple Ensemble ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 多尺度VNA
        vna_scales = [5, 12, 20, 30]
        vna_preds = []
        for k in vna_scales:
            nn = NNA(method='nearest', k=k, power=-2)
            nn.fit(train_df[['Lon', 'Lat']].values, train_df['Conc'].values)
            pred_grid = nn.predict(X_grid_full)
            vna_preds.append(extract_grid_pred(test_df, lon_cmaq, lat_cmaq, pred_grid, ny, nx))
        vna_pred = np.mean(vna_preds, axis=0)

        # aVNA
        train_df = train_df.copy()
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(train_df[['Lon', 'Lat']].values, train_df['bias'].values)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_pred_grid = y_grid_model_full + bias_grid
        avna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, avna_pred_grid, ny, nx)

        # eVNA
        train_df['ratio'] = train_df['Conc'] / train_df['CMAQ']
        nn_ratio = NNA(method='nearest', k=15, power=-2)
        nn_ratio.fit(train_df[['Lon', 'Lat']].values, train_df['ratio'].values)
        ratio_grid = nn_ratio.predict(X_grid_full)
        evna_pred_grid = y_grid_model_full * ratio_grid
        evna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, evna_pred_grid, ny, nx)

        # 权重
        obs_mean = train_df['Conc'].mean()
        cmaq_test = test_df['CMAQ'].values
        relative_dev = (cmaq_test - obs_mean) / obs_mean

        w_vna = np.clip(0.5 + 0.5 * relative_dev, 0.2, 0.6)
        w_avna = np.clip(0.3 - 0.3 * relative_dev, 0.2, 0.5)
        w_evna = 1 - w_vna - w_avna
        w_evna = np.clip(w_evna, 0.1, 0.4)

        mste_pred = w_vna * vna_pred + w_avna * avna_pred + w_evna * evna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'mste': mste_pred
        }

        print(f"  Fold {fold_id}: completed")

    mste_all = np.concatenate([results[f]['mste'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, mste_all)
    print(f"  MultiScale Triple Ensemble: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_power_adaptive_ensemble(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    PowerAdaptiveEnsemble: 幂自适应集成
    使用不同的power参数调整权重
    """
    print("\n=== Power Adaptive Ensemble ===")

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
        train_df = train_df.copy()
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(train_df[['Lon', 'Lat']].values, train_df['bias'].values)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_pred_grid = y_grid_model_full + bias_grid

        # 提取预测
        vna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, vna_pred_grid, ny, nx)
        avna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, avna_pred_grid, ny, nx)

        obs_mean = train_df['Conc'].mean()
        cmaq_test = test_df['CMAQ'].values

        # 相对偏差
        rel_dev = (cmaq_test - obs_mean) / obs_mean

        # 幂次变换
        w_vna = np.clip(0.5 * (1 + np.sign(rel_dev) * np.abs(rel_dev)**0.5), 0.25, 0.75)
        w_avna = 1 - w_vna

        pae_pred = w_vna * vna_pred + w_avna * avna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'pae': pae_pred
        }

        print(f"  Fold {fold_id}: completed")

    pae_all = np.concatenate([results[f]['pae'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, pae_all)
    print(f"  Power Adaptive Ensemble: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_fifth_ensemble(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    FifthEnsemble: 第五轮集成，结合所有最佳策略
    """
    print("\n=== Fifth Ensemble ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 多尺度VNA
        vna_scales = [8, 15, 25]
        vna_preds = []
        for k in vna_scales:
            nn = NNA(method='nearest', k=k, power=-2)
            nn.fit(train_df[['Lon', 'Lat']].values, train_df['Conc'].values)
            pred_grid = nn.predict(X_grid_full)
            vna_preds.append(extract_grid_pred(test_df, lon_cmaq, lat_cmaq, pred_grid, ny, nx))
        vna_pred = np.mean(vna_preds, axis=0)

        # aVNA
        train_df = train_df.copy()
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(train_df[['Lon', 'Lat']].values, train_df['bias'].values)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_pred_grid = y_grid_model_full + bias_grid
        avna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, avna_pred_grid, ny, nx)

        # eVNA
        train_df['ratio'] = train_df['Conc'] / train_df['CMAQ']
        nn_ratio = NNA(method='nearest', k=15, power=-2)
        nn_ratio.fit(train_df[['Lon', 'Lat']].values, train_df['ratio'].values)
        ratio_grid = nn_ratio.predict(X_grid_full)
        evna_pred_grid = y_grid_model_full * ratio_grid
        evna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, evna_pred_grid, ny, nx)

        obs_mean = train_df['Conc'].mean()
        obs_std = train_df['Conc'].std()
        cmaq_test = test_df['CMAQ'].values

        # Z-score偏差
        z_dev = (cmaq_test - obs_mean) / obs_std

        # 自适应权重
        w_vna = np.clip(0.5 * (1 + np.tanh(0.6 * z_dev)), 0.2, 0.65)
        w_avna = np.clip(0.3 * (1 - np.tanh(0.6 * z_dev)), 0.15, 0.5)
        w_evna = 1 - w_vna - w_avna
        w_evna = np.clip(w_evna, 0.1, 0.35)

        fe_pred = w_vna * vna_pred + w_avna * avna_pred + w_evna * evna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'fe': fe_pred
        }

        print(f"  Fold {fold_id}: completed")

    fe_all = np.concatenate([results[f]['fe'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, fe_all)
    print(f"  Fifth Ensemble: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("年平均数据高级融合方法测试 (第五轮)")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    methods = [
        ('OptimizedTripleEnsemble', run_optimized_triple_ensemble),
        ('AdaptiveTripleEnsemble', run_adaptive_triple_ensemble),
        ('MultiScaleTripleEnsemble', run_multiscale_triple_ensemble),
        ('PowerAdaptiveEnsemble', run_power_adaptive_ensemble),
        ('FifthEnsemble', run_fifth_ensemble),
    ]

    for name, func in methods:
        metrics, _ = func(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
        results[name] = metrics

    print("\n" + "="*60)
    print("第五轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/fifth_round_results.csv', index=False)

    print("\n方法排名 (按R2降序):")
    for _, row in results_df.iterrows():
        print(f"  {row['method']:25s}: R2={row['R2']:.4f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}")

    return results_df


if __name__ == '__main__':
    results_df = main()