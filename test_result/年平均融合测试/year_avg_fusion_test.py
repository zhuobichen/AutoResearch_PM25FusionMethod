"""
年平均数据融合测试
===================
测试年平均数据 vs 日数据的融合效果

方法:
1. RK-Poly: O = a + b*M + c*M^2 + GPR残差
2. ResidualKriging: O = a + b*M + GPR残差
3. eVNA: O = M * r (比率法)
4. aVNA: O = M + B (加法偏差)
5. VNA: 纯空间插值
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
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
    """
    加载并准备年平均数据
    1. 监测数据: 所有站点所有天的日均值，再做年平均
    2. CMAQ数据: 所有天的年平均
    """
    print("=== Loading and Preparing Year-Average Data ===")

    # 加载监测数据
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    # 年平均监测数据 (所有站点平均)
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
    cmaq_year_avg = pred_pm25.mean(axis=0)  # (127, 172)
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

    print(f"Final dataset: {len(monitor_year_avg)} sites with complete data")
    print(f"Monitor Conc range: [{monitor_year_avg['Conc'].min():.2f}, {monitor_year_avg['Conc'].max():.2f}]")
    print(f"CMAQ range: [{monitor_year_avg['CMAQ'].min():.2f}, {monitor_year_avg['CMAQ'].max():.2f}]")

    return monitor_year_avg, lon_cmaq, lat_cmaq, cmaq_year_avg


def run_rk_poly_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    RK-Poly融合: 多项式偏差校正 + GPR残差空间插值
    O = a + b*M + c*M^2
    """
    print("\n=== RK-Poly Fusion ===")
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        X_train = train_df[['Lon', 'Lat']].values
        X_test = test_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values
        y_test = test_df['Conc'].values
        m_train = train_df['CMAQ'].values
        m_test = test_df['CMAQ'].values

        # 二次多项式OLS
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols_poly = LinearRegression()
        ols_poly.fit(m_train_poly, y_train)
        pred_poly = ols_poly.predict(m_test_poly)
        residual_poly = y_train - ols_poly.predict(m_train_poly)

        # GPR on residuals (optimized for speed)
        gpr_poly = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1.0, normalize_y=True)
        gpr_poly.fit(X_train, residual_poly)
        gpr_poly_pred, _ = gpr_poly.predict(X_test, return_std=True)

        # 融合预测
        rk_poly_pred = pred_poly + gpr_poly_pred

        results[fold_id] = {
            'y_true': y_test,
            'rk_poly': rk_poly_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    rk_poly_all = np.concatenate([results[f]['rk_poly'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, rk_poly_all)
    print(f"  RK-Poly: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_residual_kriging_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    ResidualKriging融合: 线性偏差校正 + GPR残差
    O = a + b*M
    """
    print("\n=== ResidualKriging Fusion ===")
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 线性OLS
        ols_linear = LinearRegression()
        ols_linear.fit(train_df['CMAQ'].values.reshape(-1, 1), train_df['Conc'].values)
        pred_linear = ols_linear.intercept_ + ols_linear.coef_[0] * test_df['CMAQ'].values
        residual_linear = train_df['Conc'].values - (ols_linear.intercept_ + ols_linear.coef_[0] * train_df['CMAQ'].values)

        # GPR on residuals (optimized for speed)
        gpr_linear = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1.0, normalize_y=True)
        gpr_linear.fit(train_df[['Lon', 'Lat']].values, residual_linear)
        gpr_linear_pred, _ = gpr_linear.predict(test_df[['Lon', 'Lat']].values, return_std=True)

        # 融合预测
        rk_pred = pred_linear + gpr_linear_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'rk': rk_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    rk_all = np.concatenate([results[f]['rk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, rk_all)
    print(f"  ResidualKriging: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_evna_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    eVNA融合: 比率法
    O = M * r (r = O/M的空间插值)
    """
    print("\n=== eVNA Fusion ===")

    # 创建网格坐标
    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 计算比率 r = O/M
        train_df = train_df.copy()
        train_df['ratio'] = train_df['Conc'] / train_df['CMAQ']

        # NNA空间插值比率
        nn = NNA(method='nearest', k=15, power=-2)
        nn.fit(train_df[['Lon', 'Lat']].values, train_df['ratio'].values)

        # 预测网格上的比率
        ratio_grid = nn.predict(X_grid_full)

        # eVNA预测: M * r
        evna_pred_grid = y_grid_model_full * ratio_grid

        # 提取测试站点预测值
        evna_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            row, col = idx // nx, idx % nx
            evna_pred[i] = evna_pred_grid[row * nx + col]

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'evna': evna_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    evna_all = np.concatenate([results[f]['evna'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, evna_all)
    print(f"  eVNA: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_avna_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    aVNA融合: 加法偏差法
    O = M + B (B = O-M的空间插值)
    """
    print("\n=== aVNA Fusion ===")

    # 创建网格坐标
    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # 计算偏差 B = O-M
        train_df = train_df.copy()
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']

        # NNA空间插值偏差
        nn = NNA(method='nearest', k=15, power=-2)
        nn.fit(train_df[['Lon', 'Lat']].values, train_df['bias'].values)

        # 预测网格上的偏差
        bias_grid = nn.predict(X_grid_full)

        # aVNA预测: M + B
        avna_pred_grid = y_grid_model_full + bias_grid

        # 提取测试站点预测值
        avna_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            row, col = idx // nx, idx % nx
            avna_pred[i] = avna_pred_grid[row * nx + col]

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'avna': avna_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    avna_all = np.concatenate([results[f]['avna'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, avna_all)
    print(f"  aVNA: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_vna_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    VNA融合: 纯空间插值
    直接对观测值进行空间插值，不使用模型数据
    """
    print("\n=== VNA Fusion ===")

    # 创建网格坐标
    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        # NNA空间插值观测值
        nn = NNA(method='nearest', k=15, power=-2)
        nn.fit(train_df[['Lon', 'Lat']].values, train_df['Conc'].values)

        # 预测网格上的值
        vna_pred_grid = nn.predict(X_grid_full)

        # 提取测试站点预测值
        vna_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - test_df.iloc[i]['Lon'])**2 + (lat_cmaq - test_df.iloc[i]['Lat'])**2)
            idx = np.argmin(dist)
            row, col = idx // nx, idx % nx
            vna_pred[i] = vna_pred_grid[row * nx + col]

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'vna': vna_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    vna_all = np.concatenate([results[f]['vna'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, vna_all)
    print(f"  VNA: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("年平均数据融合测试")
    print("="*60)

    # Step 1: 加载并准备年平均数据
    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    # Step 2: 运行各种融合方法
    results = {}

    # RK-Poly
    rk_poly_metrics, _ = run_rk_poly_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
    results['RK-Poly'] = rk_poly_metrics

    # ResidualKriging
    rk_metrics, _ = run_residual_kriging_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
    results['ResidualKriging'] = rk_metrics

    # eVNA
    evna_metrics, _ = run_evna_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
    results['eVNA'] = evna_metrics

    # aVNA
    avna_metrics, _ = run_avna_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
    results['aVNA'] = avna_metrics

    # VNA
    vna_metrics, _ = run_vna_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
    results['VNA'] = vna_metrics

    # Step 3: 保存结果
    print("\n" + "="*60)
    print("融合结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/results.csv', index=False)

    print("\n方法排名 (按R2降序):")
    for _, row in results_df.iterrows():
        print(f"  {row['method']:20s}: R2={row['R2']:.4f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}, MB={row['MB']:.2f}")

    # 最佳方法
    best_method = results_df.iloc[0]['method']
    best_r2 = results_df.iloc[0]['R2']
    print(f"\n最佳方法: {best_method} (R2={best_r2:.4f})")

    print(f"\n结果已保存到: {output_dir}/results.csv")

    return results_df


if __name__ == '__main__':
    results_df = main()