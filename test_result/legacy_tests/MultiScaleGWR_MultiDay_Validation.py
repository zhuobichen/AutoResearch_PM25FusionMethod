"""
MultiScaleGWR Multi-Day Validation
==================================
对多日数据进行十折交叉验证
比较MultiScaleGWR和RK-Poly方法

日期: 2020-01-01 至 2020-01-05
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import netCDF4 as nc

root_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
cmaq_file = f'{root_dir}/test_data/raw/CMAQ/2020_PM25.nc'
monitor_file = f'{root_dir}/test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv'
fold_file = f'{root_dir}/test_data/fold_split_table.csv'
output_dir = f'{root_dir}/test_result'
os.makedirs(output_dir, exist_ok=True)


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics"""
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
    """Get CMAQ value at site using nearest neighbor"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def run_multiscale_gwr_ten_fold(day_df, bandwidths=[0.5, 1.0, 2.0]):
    """
    MultiScaleGWR十折交叉验证

    公式:
    - 局部回归: y = β₀ + β₁ * M
    - 权重: w = exp(-d²/(2h²))
    - 多尺度融合3个带宽的预测
    """
    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        coords_train = train_df[['Lon', 'Lat']].values
        m_train = train_df['CMAQ'].values
        y_train = train_df['Conc'].values

        coords_test = test_df[['Lon', 'Lat']].values
        m_test = test_df['CMAQ'].values

        n_scales = len(bandwidths)
        nn_finder = NearestNeighbors(n_neighbors=min(20, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        ms_pred = np.zeros((len(test_df), n_scales))

        for b_idx, bandwidth in enumerate(bandwidths):
            for i in range(len(test_df)):
                dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(15, len(coords_train)))

                weights = np.exp(-0.5 * (dists[0] / bandwidth)**2)
                weights = weights / (weights.sum() + 1e-6)

                X_local = np.column_stack([np.ones(len(indices[0])), m_train[indices[0]]])
                y_local = y_train[indices[0]]

                W = np.diag(weights)
                try:
                    XTWX = X_local.T @ W @ X_local + np.eye(2) * 1e-6
                    XTWy = X_local.T @ W @ y_local
                    beta = np.linalg.solve(XTWX, XTWy)
                    ms_pred[i, b_idx] = beta[0] + beta[1] * m_test[i]
                except:
                    ms_pred[i, b_idx] = y_train.mean() + (m_test[i] - m_train.mean()) * np.corrcoef(y_train, m_train)[0, 1]

        # 多尺度融合 (等权重平均)
        if len(bandwidths) == 3:
            final_pred = (ms_pred[:, 0] + ms_pred[:, 1] + ms_pred[:, 2]) / 3.0
        else:
            scale_weights = np.ones(n_scales) / n_scales
            final_pred = np.sum(ms_pred * np.array(scale_weights).reshape(1, -1), axis=1)

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'msgwr_pred': final_pred,
            'ms_pred': ms_pred
        }

    return results


def run_rk_poly_ten_fold(day_df):
    """
    RK-Poly (Polynomial Residual Kriging) 十折交叉验证
    使用二次多项式OLS + IDW残差插值 (快速版)
    """
    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

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

        # IDW残差插值 (快速版)
        nn_finder = NearestNeighbors(n_neighbors=min(15, len(X_train)), algorithm='ball_tree')
        nn_finder.fit(X_train)

        residual_pred = np.zeros(len(X_test))
        for i in range(len(X_test)):
            dists, indices = nn_finder.kneighbors([X_test[i]], n_neighbors=min(10, len(X_train)))
            weights = 1.0 / (dists[0] + 0.1)
            weights = weights / weights.sum()
            residual_pred[i] = np.sum(weights * residual_poly[indices[0]])

        # 融合预测
        rk_poly_pred = pred_poly + residual_pred

        results[fold_id] = {
            'y_true': y_test,
            'rk_poly_pred': rk_poly_pred
        }

    return results


def process_single_day(selected_day, monitor_df, fold_df, lon_cmaq, lat_cmaq, pred_pm25):
    """处理单日数据，返回MultiScaleGWR和RK-Poly的十折结果"""

    from datetime import datetime
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    pred_day = pred_pm25[day_idx]

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    # 提取站点CMAQ值
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values
    day_df = day_df.dropna(subset=['CMAQ', 'fold'])

    print(f"\n  Date: {selected_day}, Sites: {len(day_df)}")

    # MultiScaleGWR
    msgwr_results = run_multiscale_gwr_ten_fold(day_df, bandwidths=[0.5, 1.0, 2.0])

    # RK-Poly
    rkpoly_results = run_rk_poly_ten_fold(day_df)

    return day_df, msgwr_results, rkpoly_results


def main():
    print("="*70)
    print("MultiScaleGWR Multi-Day 十折交叉验证")
    print("="*70)

    # 加载数据
    print("\n=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    print(f"Monitor data loaded: {len(monitor_df)} records")
    print(f"CMAQ data shape: {pred_pm25.shape}")

    # 测试日期
    test_dates = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']

    # 存储所有结果
    all_results = []
    all_fold_details = []

    for selected_day in test_dates:
        print(f"\n{'='*60}")
        print(f"Processing: {selected_day}")
        print(f"{'='*60}")

        day_df, msgwr_results, rkpoly_results = process_single_day(
            selected_day, monitor_df, fold_df, lon_cmaq, lat_cmaq, pred_pm25
        )

        # === MultiScaleGWR 汇总 ===
        msgwr_all = np.concatenate([msgwr_results[f]['msgwr_pred'] for f in range(1, 11) if msgwr_results[f]])
        true_all = np.concatenate([msgwr_results[f]['y_true'] for f in range(1, 11) if msgwr_results[f]])

        msgwr_metrics = compute_metrics(true_all, msgwr_all)

        # 每折Metrics (MultiScaleGWR)
        for fold_id in range(1, 11):
            if msgwr_results[fold_id]:
                fold_true = msgwr_results[fold_id]['y_true']
                fold_pred = msgwr_results[fold_id]['msgwr_pred']
                fold_metrics = compute_metrics(fold_true, fold_pred)
                all_fold_details.append({
                    'Date': selected_day,
                    'Method': 'MultiScaleGWR',
                    'Fold': fold_id,
                    **fold_metrics
                })

        # === RK-Poly 汇总 ===
        rkpoly_all = np.concatenate([rkpoly_results[f]['rk_poly_pred'] for f in range(1, 11) if rkpoly_results[f]])

        rkpoly_metrics = compute_metrics(true_all, rkpoly_all)

        # 每折Metrics (RK-Poly)
        for fold_id in range(1, 11):
            if rkpoly_results[fold_id]:
                fold_true = rkpoly_results[fold_id]['y_true']
                fold_pred = rkpoly_results[fold_id]['rk_poly_pred']
                fold_metrics = compute_metrics(fold_true, fold_pred)
                all_fold_details.append({
                    'Date': selected_day,
                    'Method': 'RK-Poly',
                    'Fold': fold_id,
                    **fold_metrics
                })

        print(f"\n  MultiScaleGWR: R2={msgwr_metrics['R2']:.4f}, MAE={msgwr_metrics['MAE']:.2f}, RMSE={msgwr_metrics['RMSE']:.2f}")
        print(f"  RK-Poly:       R2={rkpoly_metrics['R2']:.4f}, MAE={rkpoly_metrics['MAE']:.2f}, RMSE={rkpoly_metrics['RMSE']:.2f}")

        # 检查异常折
        print("\n  Checking abnormal folds (negative R2)...")
        for fold_id in range(1, 11):
            if msgwr_results[fold_id]:
                fold_pred = msgwr_results[fold_id]['msgwr_pred']
                fold_true = msgwr_results[fold_id]['y_true']
                fold_r2 = compute_metrics(fold_true, fold_pred)['R2']
                if fold_r2 < 0:
                    print(f"    WARNING: MultiScaleGWR Fold {fold_id} has negative R2 = {fold_r2:.4f}")

        all_results.append({
            'Date': selected_day,
            'Method': 'MultiScaleGWR',
            **msgwr_metrics
        })
        all_results.append({
            'Date': selected_day,
            'Method': 'RK-Poly',
            **rkpoly_metrics
        })

    # === 保存汇总结果 ===
    results_df = pd.DataFrame(all_results)
    results_df = results_df[['Date', 'Method', 'R2', 'MAE', 'RMSE', 'MB']]
    results_df.to_csv(f'{output_dir}/MultiScaleGWR_multi_day_results.csv', index=False)
    print(f"\n\nResults saved to: {output_dir}/MultiScaleGWR_multi_day_results.csv")

    # === 保存每折详情 ===
    fold_df_details = pd.DataFrame(all_fold_details)
    fold_df_details = fold_df_details[['Date', 'Method', 'Fold', 'R2', 'MAE', 'RMSE', 'MB']]
    fold_df_details.to_csv(f'{output_dir}/MultiScaleGWR_multi_day_fold_details.csv', index=False)

    # === 生成报告 ===
    report = generate_report(results_df, fold_df_details, test_dates)
    with open(f'{output_dir}/MultiScaleGWR_multi_day_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to: {output_dir}/MultiScaleGWR_multi_day_report.md")

    # === 打印汇总表 ===
    print("\n" + "="*70)
    print("MultiScaleGWR vs RK-Poly 对比汇总")
    print("="*70)
    print(f"\n{'Date':<12} {'Method':<15} {'R2':>8} {'MAE':>8} {'RMSE':>8}")
    print("-"*55)
    for _, row in results_df.iterrows():
        print(f"{row['Date']:<12} {row['Method']:<15} {row['R2']:>8.4f} {row['MAE']:>8.2f} {row['RMSE']:>8.2f}")

    # === 异常折汇总 ===
    print("\n" + "="*70)
    print("异常折检查 (负R2)")
    print("="*70)
    abnormal_folds = fold_df_details[fold_df_details['R2'] < 0]
    if len(abnormal_folds) > 0:
        for _, row in abnormal_folds.iterrows():
            print(f"  {row['Date']} | {row['Method']} | Fold {row['Fold']}: R2={row['R2']:.4f}")
    else:
        print("  无异常折 (所有折R2均 >= 0)")

    return results_df, fold_df_details


def generate_report(results_df, fold_df_details, test_dates):
    """生成Markdown报告"""

    report = """# MultiScaleGWR Multi-Day Validation Report

## 实验设置

- **方法**: MultiScaleGWR (多尺度地理加权回归)
- **公式**:
  - 局部回归: y = β₀ + β₁ * M
  - 权重: w = exp(-d²/(2h²))
  - 多尺度融合3个带宽(0.5, 1.0, 2.0)的预测

- **对比方法**: RK-Poly (多项式残差克里金)
- **验证方式**: 十折交叉验证

## 测试日期

"""

    for d in test_dates:
        report += f"- {d}\n"

    report += """
## 汇总结果

| Date | Method | R2 | MAE | RMSE | MB |
|------|--------|-----|-----|------|-----|
"""

    for _, row in results_df.iterrows():
        report += f"| {row['Date']} | {row['Method']} | {row['R2']:.4f} | {row['MAE']:.2f} | {row['RMSE']:.2f} | {row['MB']:.2f} |\n"

    # MultiScaleGWR vs RK-Poly 对比
    report += """
## 方法对比

"""

    for date in test_dates:
        date_results = results_df[results_df['Date'] == date]
        msgwr = date_results[date_results['Method'] == 'MultiScaleGWR'].iloc[0]
        rkpoly = date_results[date_results['Method'] == 'RK-Poly'].iloc[0]

        r2_diff = msgwr['R2'] - rkpoly['R2']
        better = "MultiScaleGWR" if r2_diff > 0 else "RK-Poly"

        report += f"""### {date}

- **MultiScaleGWR**: R2={msgwr['R2']:.4f}, MAE={msgwr['MAE']:.2f}, RMSE={msgwr['RMSE']:.2f}
- **RK-Poly**: R2={rkpoly['R2']:.4f}, MAE={rkpoly['MAE']:.2f}, RMSE={rkpoly['RMSE']:.2f}
- **更优方法**: {better} (R2差异: {r2_diff:+.4f})

"""

    # 异常折检查
    report += """
## 异常折检查

"""
    abnormal_folds = fold_df_details[fold_df_details['R2'] < 0]
    if len(abnormal_folds) > 0:
        report += "| Date | Method | Fold | R2 | MAE | RMSE |\n"
        report += "|------|--------|------|-----|-----|------|\n"
        for _, row in abnormal_folds.iterrows():
            report += f"| {row['Date']} | {row['Method']} | {row['Fold']} | {row['R2']:.4f} | {row['MAE']:.2f} | {row['RMSE']:.2f} |\n"
    else:
        report += "无异常折。所有折的R²均大于等于0。\n"

    # 每折详情
    report += """
## 每折详细结果

### MultiScaleGWR

| Date | Fold | R2 | MAE | RMSE |
|------|------|-----|-----|------|
"""
    msgwr_folds = fold_df_details[fold_df_details['Method'] == 'MultiScaleGWR']
    for _, row in msgwr_folds.iterrows():
        report += f"| {row['Date']} | {row['Fold']} | {row['R2']:.4f} | {row['MAE']:.2f} | {row['RMSE']:.2f} |\n"

    report += """
### RK-Poly

| Date | Fold | R2 | MAE | RMSE |
|------|------|-----|-----|------|
"""
    rkpoly_folds = fold_df_details[fold_df_details['Method'] == 'RK-Poly']
    for _, row in rkpoly_folds.iterrows():
        report += f"| {row['Date']} | {row['Fold']} | {row['R2']:.4f} | {row['MAE']:.2f} | {row['RMSE']:.2f} |\n"

    report += """
## 结论

"""

    # 计算平均R2
    msgwr_avg_r2 = results_df[results_df['Method'] == 'MultiScaleGWR']['R2'].mean()
    rkpoly_avg_r2 = results_df[results_df['Method'] == 'RK-Poly']['R2'].mean()

    report += f"- MultiScaleGWR 平均R²: {msgwr_avg_r2:.4f}\n"
    report += f"- RK-Poly 平均R²: {rkpoly_avg_r2:.4f}\n"

    if msgwr_avg_r2 > rkpoly_avg_r2:
        report += f"- MultiScaleGWR 在多日测试中整体表现更优 (R²高出 {msgwr_avg_r2 - rkpoly_avg_r2:.4f})\n"
    else:
        report += f"- RK-Poly 在多日测试中整体表现更优 (R²高出 {rkpoly_avg_r2 - msgwr_avg_r2:.4f})\n"

    return report


if __name__ == '__main__':
    results_df, fold_df_details = main()
    print("\n\nDone!")