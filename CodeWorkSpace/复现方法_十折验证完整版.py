"""
PM2.5融合方法十折验证完整版
==========================
对所有复现方法进行十折交叉验证

运行方式:
python CodeWorkSpace/复现方法_十折验证完整版.py
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import netCDF4 as nc
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/CodeWorkSpace/复现方法代码')

# 导入方法
from ReproductionMethods import REPRODUCTION_METHODS

root_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
cmaq_file = f'{root_dir}/test_data/raw/CMAQ/2020_PM25.nc'
monitor_file = f'{root_dir}/test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv'
fold_file = f'{root_dir}/test_data/fold_split_table.csv'
output_dir = f'{root_dir}/test_result/复现方法'

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

# 加载CMAQ数据
ds = nc.Dataset(cmaq_file, 'r')
lon_cmaq = ds.variables['lon'][:]
lat_cmaq = ds.variables['lat'][:]
pred_pm25 = ds.variables['pred_PM25'][:]
ds.close()

# 提取指定日期的CMAQ数据
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

# 获取CMAQ网格坐标
ny, nx = lon_cmaq.shape
X_grid_full = np.column_stack([
    lon_cmaq.ravel(),
    lat_cmaq.ravel()
])
y_model_grid_full = pred_day.ravel()

# 创建用于获取网格值的函数
def get_grid_value_at_lonlat(lon, lat, lon_grid, lat_grid, values_grid):
    """获取网格点在指定经纬度的值（最近邻）"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    return values_grid[idx]

# 定义要测试的方法
def get_fusion_methods():
    """获取所有需要测试的方法"""
    methods = {}

    # 基准方法
    methods['CMAQ'] = None

    # 复现方法
    for name, cls in REPRODUCTION_METHODS.items():
        methods[f'Reproduce_{name}'] = cls

    # 已有创新方法
    try:
        from CodeWorkSpace.新融合方法代码.SuperEnsemble import SuperStackingEnsemble
        methods['SuperStackingEnsemble'] = SuperStackingEnsemble
    except:
        pass

    return methods

# 十折交叉验证
def run_10fold_cv():
    """运行十折交叉验证"""
    methods = get_fusion_methods()
    print(f"\n=== 测试方法列表 ({len(methods)}) ===")
    for name in methods.keys():
        print(f"  - {name}")

    all_results = {}

    for fold_id in range(1, 11):
        print(f"\n{'='*60}")
        print(f"Fold {fold_id}/10")
        print('='*60)

        # 训练集和验证集
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        print(f"训练站点：{len(train_df)}, 验证站点：{len(test_df)}")

        # 准备训练数据
        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        X_train = train_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values
        y_model_train = train_df['CMAQ'].values

        X_test = test_df[['Lon', 'Lat']].values
        y_test = test_df['Conc'].values
        y_model_test = test_df['CMAQ'].values

        # 在整个网格上预测
        print("  在整个网格上预测...")
        y_grid_pred_all = {}

        for method_name, method_cls in methods.items():
            if method_name == 'CMAQ':
                y_grid_pred_all['CMAQ'] = y_model_grid_full.copy()
            else:
                if method_cls is None:
                    continue
                try:
                    model = method_cls()
                    model.fit(X_train, y_train, y_model_train)
                    y_grid_pred = model.predict(X_grid_full, y_model_grid_full)
                    y_grid_pred_all[method_name] = y_grid_pred
                except Exception as e:
                    print(f"  [错误] {method_name}: {str(e)[:50]}")
                    continue

        # 在验证站点位置提取预测值
        for method_name in y_grid_pred_all.keys():
            y_pred = np.zeros(len(test_df))
            for i, (_, row) in enumerate(test_df.iterrows()):
                y_pred[i] = get_grid_value_at_lonlat(
                    row['Lon'], row['Lat'],
                    lon_cmaq, lat_cmaq,
                    y_grid_pred_all[method_name]
                )

            # 计算指标
            metrics = compute_metrics(y_test, y_pred)

            if method_name not in all_results:
                all_results[method_name] = {
                    'y_true': [],
                    'y_pred': []
                }

            all_results[method_name]['y_true'].extend(y_test)
            all_results[method_name]['y_pred'].extend(y_pred)

            print(f"  {method_name:30s}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return all_results

# 运行测试
print("\n" + "="*60)
print("开始十折交叉验证")
print("="*60)

results = run_10fold_cv()

# 汇总结果
print("\n" + "="*60)
print("十折交叉验证汇总")
print("="*60)

summary_data = []
for method_name in sorted(results.keys(), key=lambda x: -compute_metrics(
    np.array(results[x]['y_true']),
    np.array(results[x]['y_pred'])
)['R2']):

    y_true = np.array(results[method_name]['y_true'])
    y_pred = np.array(results[method_name]['y_pred'])
    metrics = compute_metrics(y_true, y_pred)

    summary_data.append({
        'method': method_name,
        'R2': metrics['R2'],
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'MB': metrics['MB']
    })

    print(f"{method_name:35s}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

# 保存结果
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f'{output_dir}/reproduction_summary.csv', index=False)
print(f"\n结果已保存到：{output_dir}/reproduction_summary.csv")

# 返回结果供后续使用
print("\n=== 测试完成 ===")
print(f"共测试 {len(results)} 个方法")