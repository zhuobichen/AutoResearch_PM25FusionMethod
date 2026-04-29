"""
十折交叉验证测试
================
对VNA/eVNA/aVNA方法进行十折交叉验证
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import netCDF4 as nc
from tqdm import tqdm

# 添加路径
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')
from Code.VNAeVNAaVNA import VNAFusion
from Code.VNAeVNAaVNA.core import VNAFusionCore
from Code.VNAeVNAaVNA.nna_methods import NNA

root_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
cmaq_file = f'{root_dir}/test_data/raw/CMAQ/2020_PM25.nc'
monitor_file = f'{root_dir}/test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv'
fold_file = f'{root_dir}/test_data/fold_split_table.csv'
output_dir = f'{root_dir}/test_result/基准方法'

os.makedirs(output_dir, exist_ok=True)

# 加载数据
print("=== 加载数据 ===")
monitor_df = pd.read_csv(monitor_file)
fold_df = pd.read_csv(fold_file)
print(f"监测数据：{len(monitor_df)} 条")
print(f"站点数：{monitor_df['Site'].nunique()}")

# 筛选指定日期
selected_day = '2020-01-01'
day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
print(f"日期 {selected_day} 数据：{len(day_df)} 条")

# 合并十折信息
day_df = day_df.merge(fold_df, on='Site', how='left')
print(f"合并后数据：{len(day_df)} 条")

# 加载CMAQ数据获取站点位置的模型值
ds = nc.Dataset(cmaq_file, 'r')
lon_cmaq = ds.variables['lon'][:]
lat_cmaq = ds.variables['lat'][:]
pred_pm25 = ds.variables['pred_PM25'][:]
ds.close()

# 找到指定日期的索引（time是0-364的整数，对应2020年第几天）
from datetime import datetime
date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
day_idx = (date_obj - datetime(2020, 1, 1)).days
pred_day = pred_pm25[day_idx]

# 提取站点位置的CMAQ值
def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    """获取站点位置的CMAQ值（最近邻）"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]

print("\n=== 提取站点CMAQ值 ===")
cmaq_values = []
for _, row in day_df.iterrows():
    val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
    cmaq_values.append(val)
day_df['CMAQ'] = cmaq_values

print(f"CMAQ值范围：{day_df['CMAQ'].min():.2f} ~ {day_df['CMAQ'].max():.2f}")
print(f"监测值范围：{day_df['Conc'].min():.2f} ~ {day_df['Conc'].max():.2f}")

# 计算metrics
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

# 十折交叉验证
print("\n=== 十折交叉验证 ===")
methods = ['CMAQ', 'VNA', 'aVNA', 'eVNA']
results = {m: [] for m in methods}

for fold_id in range(1, 11):
    print(f"\n--- Fold {fold_id} ---")

    # 训练集和验证集
    train_df = day_df[day_df['fold'] != fold_id].copy()
    test_df = day_df[day_df['fold'] == fold_id].copy()

    print(f"训练站点：{len(train_df)}, 验证站点：{len(test_df)}")

    # 准备训练数据（过滤NaN）
    train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
    train_df['x'] = train_df['Lon']
    train_df['y'] = train_df['Lat']
    train_df['mod'] = train_df['CMAQ']
    train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
    train_df['r_n'] = train_df['Conc'] / train_df['CMAQ']

    # 验证数据也要过滤
    test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

    # 1. CMAQ（直接用模型值）
    results['CMAQ'].append({
        'fold': fold_id,
        'y_true': test_df['Conc'].values,
        'y_pred': test_df['CMAQ'].values
    })

    # 2. VNA/aVNA/eVNA
    nn = NNA(method='voronoi', k=30, power=-2)
    nn.fit(
        train_df[['x', 'y']],
        train_df[['Conc', 'mod', 'bias', 'r_n']]
    )

    # ===== 正确流程：先在整个网格上插值，再在验证站点位置提取 =====
    # 创建CMAQ网格坐标
    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([
        lon_cmaq.ravel(),
        lat_cmaq.ravel()
    ])

    # 在整个网格上预测（插值）
    zdf_grid = nn.predict(X_grid_full, njobs=4)
    vna_grid = zdf_grid[:, 0]  # VNA在网格上的值
    vna_bias_grid = zdf_grid[:, 2]  # bias在网格上的值
    vna_rn_grid = zdf_grid[:, 3]  # rn在网格上的值

    # 在验证站点位置提取预测值
    # 用最近邻提取（与CMAQ提取方式一致）
    vna_pred = np.zeros(len(test_df))
    vna_bias_pred = np.zeros(len(test_df))
    vna_rn_pred = np.zeros(len(test_df))
    for i, (_, row) in enumerate(test_df.iterrows()):
        dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
        idx = np.argmin(dist)
        vna_pred[i] = vna_grid[idx]
        vna_bias_pred[i] = vna_bias_grid[idx]
        vna_rn_pred[i] = vna_rn_grid[idx]

    # aVNA = CMAQ + bias
    aVNA_pred = test_df['CMAQ'].values + vna_bias_pred
    # eVNA = CMAQ * rn
    eVNA_pred = test_df['CMAQ'].values * vna_rn_pred

    results['VNA'].append({
        'fold': fold_id,
        'y_true': test_df['Conc'].values,
        'y_pred': vna_pred
    })
    results['aVNA'].append({
        'fold': fold_id,
        'y_true': test_df['Conc'].values,
        'y_pred': aVNA_pred
    })
    results['eVNA'].append({
        'fold': fold_id,
        'y_true': test_df['Conc'].values,
        'y_pred': eVNA_pred
    })

    # 打印本折结果
    for m in methods:
        y_true = results[m][-1]['y_true']
        y_pred = results[m][-1]['y_pred']
        metrics = compute_metrics(y_true, y_pred)
        print(f"  {m}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

# 汇总结果
print("\n" + "="*60)
print("十折交叉验证汇总")
print("="*60)

summary = {}
for m in methods:
    all_true = np.concatenate([r['y_true'] for r in results[m]])
    all_pred = np.concatenate([r['y_pred'] for r in results[m]])
    metrics = compute_metrics(all_true, all_pred)
    summary[m] = metrics
    print(f"{m:>8}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

# 保存结果
summary_df = pd.DataFrame(summary).T
summary_df.to_csv(f'{output_dir}/benchmark_summary.csv')
print(f"\n结果已保存到：{output_dir}/benchmark_summary.csv")
