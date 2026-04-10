"""
V6-Ensemble-V1: 年平均数据最佳融合方法
======================================

方法: 基于CMAQ偏差引导的三重自适应集成

原理:
- 当CMAQ值接近观测均值时，更信任VNA空间插值
- 当CMAQ偏离观测均值时，更信任aVNA/eVNA校正
- 权重根据相对偏差动态调整

性能: R2=0.5803, MAE=4.29, RMSE=9.82
      (相比VNA基准提升0.0103)
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


def run_v6_ensemble_v1(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    V6-Ensemble-V1: 基于相对偏差的线性组合

    权重公式:
    w_vna = clip(0.58 + 0.48 * relative_dev, 0.2, 0.65)
    w_avna = clip(0.22 - 0.22 * relative_dev, 0.15, 0.5)
    w_evna = clip(1 - w_vna - w_avna, 0.1, 0.35)
    """
    print("\n=== V6-Ensemble-V1 ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # ========== VNA ==========
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(train_df[['Lon', 'Lat']].values, train_df['Conc'].values)
        vna_pred_grid = nn_vna.predict(X_grid_full)

        # ========== aVNA ==========
        train_df = train_df.copy()
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(train_df[['Lon', 'Lat']].values, train_df['bias'].values)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_pred_grid = y_grid_model_full + bias_grid

        # ========== eVNA ==========
        train_df['ratio'] = train_df['Conc'] / train_df['CMAQ']
        nn_ratio = NNA(method='nearest', k=15, power=-2)
        nn_ratio.fit(train_df[['Lon', 'Lat']].values, train_df['ratio'].values)
        ratio_grid = nn_ratio.predict(X_grid_full)
        evna_pred_grid = y_grid_model_full * ratio_grid

        # ========== 提取预测 ==========
        vna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, vna_pred_grid, ny, nx)
        avna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, avna_pred_grid, ny, nx)
        evna_pred = extract_grid_pred(test_df, lon_cmaq, lat_cmaq, evna_pred_grid, ny, nx)

        # ========== 自适应权重 ==========
        obs_mean = train_df['Conc'].mean()
        cmaq_test = test_df['CMAQ'].values
        relative_dev = (cmaq_test - obs_mean) / obs_mean

        # 核心权重公式
        w_vna = np.clip(0.58 + 0.48 * relative_dev, 0.2, 0.65)
        w_avna = np.clip(0.22 - 0.22 * relative_dev, 0.15, 0.5)
        w_evna = 1 - w_vna - w_avna
        w_evna = np.clip(w_evna, 0.1, 0.35)

        # ========== 融合预测 ==========
        v1_pred = w_vna * vna_pred + w_avna * avna_pred + w_evna * evna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'v1': v1_pred
        }

        print(f"  Fold {fold_id}: completed")

    v1_all = np.concatenate([results[f]['v1'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, v1_all)
    print(f"  V6-Ensemble-V1: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("V6-Ensemble-V1: 年平均数据最佳融合方法")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    # 运行V6-Ensemble-V1
    metrics, _ = run_v6_ensemble_v1(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)

    # 保存结果
    results_df = pd.DataFrame([{'method': 'V6-Ensemble-V1', **metrics}])
    results_df.to_csv(f'{output_dir}/v6_ensemble_v1_results.csv', index=False)

    print("\n" + "="*60)
    print("结果")
    print("="*60)
    print(f"R2  = {metrics['R2']:.4f}")
    print(f"MAE = {metrics['MAE']:.2f}")
    print(f"RMSE = {metrics['RMSE']:.2f}")
    print(f"MB = {metrics['MB']:.2f}")
    print(f"\n结果已保存到: {output_dir}/v6_ensemble_v1_results.csv")

    return metrics


if __name__ == '__main__':
    metrics = main()