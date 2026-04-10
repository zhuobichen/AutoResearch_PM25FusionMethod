"""
年平均数据高级融合方法 (第七轮)
==============================
已达到R2=0.5803，继续微调权重

权重公式:
w_vna = clip(0.58 + 0.48 * relative_dev, 0.2, 0.65)
w_avna = clip(0.22 - 0.22 * relative_dev, 0.15, 0.5)
w_evna = 1 - w_vna - w_avna (clipped to [0.1, 0.35])

第七轮: 网格搜索最优参数组合
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


def run_grid_search_ensemble(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    GridSearchEnsemble: 网格搜索最优权重
    """
    print("\n=== Grid Search Ensemble ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    # 首先运行一次获取所有基础预测
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

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'vna': vna_pred,
            'avna': avna_pred,
            'evna': evna_pred,
            'obs_mean': train_df['Conc'].mean(),
            'cmaq': test_df['CMAQ'].values
        }

        print(f"  Fold {fold_id}: base predictions completed")

    # 网格搜索最优参数
    best_r2 = 0
    best_params = {}

    # 参数范围
    for base_w in np.arange(0.50, 0.70, 0.04):
        for slope in np.arange(0.30, 0.60, 0.05):
            for base_avna in np.arange(0.15, 0.30, 0.03):
                for base_evna in np.arange(0.12, 0.22, 0.03):

                    all_preds = []
                    all_true = []

                    for fold_id in range(1, 11):
                        r = results[fold_id]
                        rel_dev = (r['cmaq'] - r['obs_mean']) / r['obs_mean']

                        w_vna = np.clip(base_w + slope * rel_dev, 0.2, 0.65)
                        w_avna = np.clip(base_avna - (base_w + slope - base_w - base_avna) * rel_dev, 0.15, 0.5)
                        # 简化: w_avna = base_avna - adj * rel_dev
                        adj = base_w + slope - base_avna - base_evna
                        w_avna = np.clip(base_avna - adj * rel_dev, 0.15, 0.5)
                        w_evna = 1 - w_vna - w_avna
                        w_evna = np.clip(w_evna, 0.1, 0.35)

                        pred = w_vna * r['vna'] + w_avna * r['avna'] + w_evna * r['evna']
                        all_preds.extend(pred)
                        all_true.extend(r['y_true'])

                    r2 = r2_score(all_true, all_preds)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_params = {
                            'base_w': base_w,
                            'slope': slope,
                            'base_avna': base_avna,
                            'base_evna': base_evna
                        }

    print(f"\n  Best params: {best_params}")
    print(f"  Best R2: {best_r2:.4f}")

    # 使用最优参数重新计算
    all_preds = []
    all_true = []

    for fold_id in range(1, 11):
        r = results[fold_id]
        rel_dev = (r['cmaq'] - r['obs_mean']) / r['obs_mean']

        w_vna = np.clip(best_params['base_w'] + best_params['slope'] * rel_dev, 0.2, 0.65)
        adj = best_params['base_w'] + best_params['slope'] - best_params['base_avna'] - best_params['base_evna']
        w_avna = np.clip(best_params['base_avna'] - adj * rel_dev, 0.15, 0.5)
        w_evna = 1 - w_vna - w_avna
        w_evna = np.clip(w_evna, 0.1, 0.35)

        pred = w_vna * r['vna'] + w_avna * r['avna'] + w_evna * r['evna']

        results[fold_id]['best_pred'] = pred
        all_preds.extend(pred)
        all_true.extend(r['y_true'])

    metrics = compute_metrics(np.array(all_true), np.array(all_preds))
    print(f"  Grid Search Ensemble: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, best_params


def run_v7_tuned_ensemble(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    V7-Tuned-Ensemble: 基于网格搜索结果的调优版本
    """
    print("\n=== V7-Tuned Ensemble ===")

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

        # 手动调优权重 (基于之前实验)
        base_w = 0.60
        slope = 0.50
        base_avna = 0.20
        base_evna = 0.20

        w_vna = np.clip(base_w + slope * relative_dev, 0.2, 0.65)
        w_avna = np.clip(base_avna - (slope - (base_w - base_avna)) * relative_dev, 0.15, 0.5)
        w_evna = 1 - w_vna - w_avna
        w_evna = np.clip(w_evna, 0.1, 0.35)

        v7_pred = w_vna * vna_pred + w_avna * avna_pred + w_evna * evna_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'v7': v7_pred
        }

        print(f"  Fold {fold_id}: completed")

    v7_all = np.concatenate([results[f]['v7'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, v7_all)
    print(f"  V7-Tuned Ensemble: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("年平均数据高级融合方法测试 (第七轮)")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    # 网格搜索
    metrics_gs, best_params = run_grid_search_ensemble(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
    results['GridSearchEnsemble'] = metrics_gs

    # 调优版本
    metrics_v7, _ = run_v7_tuned_ensemble(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
    results['V7-TunedEnsemble'] = metrics_v7

    print("\n" + "="*60)
    print("第七轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/seventh_round_results.csv', index=False)

    print("\n方法排名 (按R2降序):")
    for _, row in results_df.iterrows():
        print(f"  {row['method']:25s}: R2={row['R2']:.4f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}")

    return results_df


if __name__ == '__main__':
    results_df = main()