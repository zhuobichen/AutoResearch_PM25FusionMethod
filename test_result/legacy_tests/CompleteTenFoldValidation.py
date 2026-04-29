"""
PM2.5融合方法完整十折验证
==========================
对所有方法进行十折交叉验证并生成最终报告
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import netCDF4 as nc
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

root_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
sys.path.insert(0, f'{root_dir}/CodeWorkSpace/复现方法代码')

from ReproductionMethods import (
    BayesianDataAssimilation, GPDownscaling, HDGC,
    UniversalKriging, IDWBiasWeighting, GenFribergFusion,
    FC1Kriging, FC2ScaledCMAQ, FCoptOptimization
)

output_dir = f'{root_dir}/test_result/复现方法'
os.makedirs(output_dir, exist_ok=True)

# 加载数据
print("=== 加载数据 ===")
monitor_df = pd.read_csv(f'{root_dir}/test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_df = pd.read_csv(f'{root_dir}/test_data/fold_split_table.csv')

selected_day = '2020-01-01'
day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
day_df = day_df.merge(fold_df, on='Site', how='left')

# 加载CMAQ数据
ds = nc.Dataset(f'{root_dir}/test_data/raw/CMAQ/2020_PM25.nc', 'r')
lon_cmaq = ds.variables['lon'][:]
lat_cmaq = ds.variables['lat'][:]
pred_pm25 = ds.variables['pred_PM25'][:]
ds.close()

date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
day_idx = (date_obj - datetime(2020, 1, 1)).days
pred_day = pred_pm25[day_idx]

def get_cmaq_at_site(lon, lat):
    dist = np.sqrt((lon_cmaq - lon)**2 + (lat_cmaq - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_cmaq.shape
    row, col = idx // nx, idx % nx
    return pred_day[row, col]

day_df['CMAQ'] = day_df.apply(lambda r: get_cmaq_at_site(r['Lon'], r['Lat']), axis=1)
print(f"总数据：{len(day_df)} 条")

# 网格
ny, nx = lon_cmaq.shape
X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
y_model_grid_full = pred_day.ravel()

# 方法定义
def get_methods():
    return [
        ('Bayesian-DA', BayesianDataAssimilation),
        ('GP-Downscaling', GPDownscaling),
        ('HDGC', HDGC),
        ('Universal-Kriging', UniversalKriging),
        ('IDW-Bias', IDWBiasWeighting),
        ('Gen-Friberg', GenFribergFusion),
        ('FC1', FC1Kriging),
        ('FC2', FC2ScaledCMAQ),
        ('FCopt', FCoptOptimization),
    ]

# 计算metrics
def compute_metrics(y_true, y_pred):
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

# 十折验证
print("\n=== 开始十折验证 ===")
methods = get_methods()
results = {name: {'y_true': [], 'y_pred': []} for name, _ in methods}

for fold_id in range(1, 11):
    print(f"\n--- Fold {fold_id}/10 ---")

    train_df = day_df[day_df['fold'] != fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
    test_df = day_df[day_df['fold'] == fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

    X_train = train_df[['Lon', 'Lat']].values
    y_train = train_df['Conc'].values
    y_model_train = train_df['CMAQ'].values

    y_test = test_df['Conc'].values

    print(f"训练: {len(train_df)}, 测试: {len(test_df)}")

    for name, cls in methods:
        try:
            model = cls()
            model.fit(X_train, y_train, y_model_train)
            y_grid_pred = model.predict(X_grid_full, y_model_grid_full)

            y_pred = np.zeros(len(test_df))
            for i, (_, row) in enumerate(test_df.iterrows()):
                dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
                idx = np.argmin(dist)
                y_pred[i] = y_grid_pred[idx]

            results[name]['y_true'].extend(y_test)
            results[name]['y_pred'].extend(y_pred)

            metrics = compute_metrics(y_test, y_pred)
            print(f"  {name:20s}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}")
        except Exception as e:
            print(f"  {name:20s}: ERROR - {str(e)[:50]}")
            results[name]['y_true'].extend(y_test)
            results[name]['y_pred'].extend([np.nan] * len(y_test))

# 汇总结果
print("\n" + "="*60)
print("十折验证汇总")
print("="*60)

summary_data = []
for name, _ in methods:
    y_true = np.array(results[name]['y_true'])
    y_pred = np.array(results[name]['y_pred'])
    metrics = compute_metrics(y_true, y_pred)

    summary_data.append({
        'method': name,
        'R2': metrics['R2'],
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'MB': metrics['MB'],
        'source': 'reproduction'
    })

    print(f"{name:25s}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

# 保存结果
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f'{output_dir}/reproduction_summary.csv', index=False)
print(f"\n结果已保存到：{output_dir}/reproduction_summary.csv")

# 读取现有排名并合并
existing_rankings = pd.read_csv(f'{root_dir}/test_result/all_methods_ranking.csv')

# 合并新结果
all_methods = pd.concat([
    existing_rankings,
    summary_df
], ignore_index=True)

# 排序
all_methods = all_methods.sort_values('R2', ascending=False)
all_methods['rank'] = range(1, len(all_methods) + 1)

# 保存完整排名
all_methods.to_csv(f'{root_dir}/test_result/all_methods_ranking_complete.csv', index=False)

print(f"\n完整排名已保存到：{root_dir}/test_result/all_methods_ranking_complete.csv")

# 显示前20名
print("\n=== 前20名方法 ===")
print(all_methods[['rank', 'method', 'R2', 'MAE', 'RMSE', 'source']].head(20).to_string(index=False))

print("\n=== 验证完成 ===")