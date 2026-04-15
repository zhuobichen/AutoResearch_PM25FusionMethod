"""
多阶段验证脚本
=============
执行阶段1(预实验5天)、阶段2(1月)、阶段3(7月)、阶段4(12月)验证

基准: eVNA
- R² >= 0.8200 (eVNA + 0.01)
- RMSE <= 12.52
- |MB| <= 0.08
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import netCDF4 as nc
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

root_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
cmaq_file = f'{root_dir}/test_data/raw/CMAQ/2020_PM25.nc'
monitor_file = f'{root_dir}/test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv'
fold_file = f'{root_dir}/test_data/fold_split_table.csv'
output_dir = f'{root_dir}/test_result/cross_day_validation'
os.makedirs(output_dir, exist_ok=True)

# 基准阈值
BENCHMARK_R2 = 0.8100
TARGET_R2 = BENCHMARK_R2 + 0.01  # 0.8200
BENCHMARK_RMSE = 12.52
BENCHMARK_MB_ABS = 0.08

# 阶段定义
STAGES = {
    'stage1': {'name': '预实验', 'days': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']},
    'stage2': {'name': '1月', 'month': 1},
    'stage3': {'name': '7月', 'month': 7},
    'stage4': {'name': '12月', 'month': 12},
}


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
    """获取CMAQ在站点的值"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def load_data(selected_day):
    """加载指定日期的数据"""
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    pred_day = pred_pm25[day_idx]

    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    return day_df, lon_cmaq, lat_cmaq


def run_rrk_ten_fold(day_df, poly_degree=2, huber_delta=1.35):
    """
    运行RRK十折交叉验证（标准模式）
    验证站点坐标 → 直接获取该位置的CMAQ值 → predict
    """
    kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
              RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

    all_y_true = []
    all_y_pred_ols = []
    all_y_pred_huber = []

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

        # 多项式特征
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        # OLS多项式校正
        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        pred_ols = ols.predict(m_test_poly)
        residual_ols = y_train - ols.predict(m_train_poly)

        # Huber稳健多项式校正
        huber = HuberRegressor(epsilon=huber_delta, max_iter=1000)
        huber.fit(m_train_poly, y_train)
        pred_huber = huber.predict(m_test_poly)
        residual_huber = y_train - huber.predict(m_train_poly)

        # GPR on residuals
        gpr_ols = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_ols.fit(X_train, residual_ols)
        gpr_ols_pred, _ = gpr_ols.predict(X_test, return_std=True)

        gpr_huber = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_huber.fit(X_train, residual_huber)
        gpr_huber_pred, _ = gpr_huber.predict(X_test, return_std=True)

        # 融合预测
        rk_ols_pred = pred_ols + gpr_ols_pred
        rk_huber_pred = pred_huber + gpr_huber_pred

        all_y_true.extend(y_test)
        all_y_pred_ols.extend(rk_ols_pred)
        all_y_pred_huber.extend(rk_huber_pred)

    all_y_true = np.array(all_y_true)
    all_y_pred_ols = np.array(all_y_pred_ols)
    all_y_pred_huber = np.array(all_y_pred_huber)

    ols_metrics = compute_metrics(all_y_true, all_y_pred_ols)
    huber_metrics = compute_metrics(all_y_true, all_y_pred_huber)

    return ols_metrics, huber_metrics


def run_single_day(selected_day):
    """在单个日期上运行十折验证"""
    print(f"\n  Processing: {selected_day}")
    day_df, lon_cmaq, lat_cmaq = load_data(selected_day)
    ols_metrics, huber_metrics = run_rrk_ten_fold(day_df)
    return {
        'day': selected_day,
        'ols': ols_metrics,
        'huber': huber_metrics
    }


def run_stage1_pre_experiment():
    """阶段1：预实验（5天）"""
    print("\n" + "="*70)
    print("阶段1：预实验（2020-01-01 ~ 2020-01-05）")
    print("="*70)
    print(f"判定条件：5天平均R² >= {TARGET_R2:.4f}")
    print(f"基准：eVNA R²={BENCHMARK_R2:.4f}, RMSE={BENCHMARK_RMSE:.2f}, |MB|={BENCHMARK_MB_ABS:.2f}")
    print("-"*70)

    days = STAGES['stage1']['days']
    all_results = []

    for day in days:
        result = run_single_day(day)
        all_results.append(result)
        print(f"    {day}: R2={result['huber']['R2']:.4f}, RMSE={result['huber']['RMSE']:.2f}, MB={result['huber']['MB']:.2f}")

    # 计算平均
    avg_r2_ols = np.mean([r['ols']['R2'] for r in all_results])
    avg_r2_huber = np.mean([r['huber']['R2'] for r in all_results])

    print(f"\n  5天平均R² (OLS): {avg_r2_ols:.4f}")
    print(f"  5天平均R² (Huber): {avg_r2_huber:.4f}")

    # 详细结果
    rows = []
    for r in all_results:
        rows.append({
            'date': r['day'],
            'R2_OLS': r['ols']['R2'],
            'RMSE_OLS': r['ols']['RMSE'],
            'MAE_OLS': r['ols']['MAE'],
            'MB_OLS': r['ols']['MB'],
            'R2_Huber': r['huber']['R2'],
            'RMSE_Huber': r['huber']['RMSE'],
            'MAE_Huber': r['huber']['MAE'],
            'MB_Huber': r['huber']['MB'],
        })

    df = pd.DataFrame(rows)
    csv_path = f'{output_dir}/预实验_5天_报告.csv'
    df.to_csv(csv_path, index=False)

    # 报告
    pass_ols = avg_r2_ols >= TARGET_R2
    pass_huber = avg_r2_huber >= TARGET_R2

    report = f"""# 预实验5天验证报告

## 基准
- eVNA: R²={BENCHMARK_R2:.4f}, RMSE={BENCHMARK_RMSE:.2f}, |MB|={BENCHMARK_MB_ABS:.2f}
- 目标: R² >= {TARGET_R2:.4f}

## 5天结果

| 日期 | R²(OLS) | RMSE(OLS) | MB(OLS) | R²(Huber) | RMSE(Huber) | MB(Huber) |
|------|----------|-----------|---------|------------|-------------|-----------|
"""
    for r in all_results:
        report += f"| {r['day']} | {r['ols']['R2']:.4f} | {r['ols']['RMSE']:.2f} | {r['ols']['MB']:.2f} | {r['huber']['R2']:.4f} | {r['huber']['RMSE']:.2f} | {r['huber']['MB']:.2f} |\n"

    report += f"""
## 判定
- 5天平均R² (OLS): {avg_r2_ols:.4f} {'>= 0.8200' if pass_ols else '< 0.8200'}
- 5天平均R² (Huber): {avg_r2_huber:.4f} {'>= 0.8200' if pass_huber else '< 0.8200'}

## 结论
- RRK-OLS: **{'通过' if pass_ols else '未通过'}**
- RRK-Huber: **{'通过' if pass_huber else '未通过'}**
"""

    report_path = f'{output_dir}/预实验_5天_报告.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n  报告已保存: {report_path}")
    print(f"  CSV已保存: {csv_path}")

    return {
        'stage': 'stage1',
        'avg_r2_ols': avg_r2_ols,
        'avg_r2_huber': avg_r2_huber,
        'pass_ols': pass_ols,
        'pass_huber': pass_huber,
        'daily_results': all_results
    }


def run_full_month(month, year=2020):
    """运行整月验证"""
    print(f"\n  Processing month {month}: ", end="", flush=True)

    if month in [1, 3, 5, 7, 8, 10, 12]:
        days_in_month = 31
    elif month in [4, 6, 9, 11]:
        days_in_month = 30
    else:
        days_in_month = 28

    all_y_true = []
    all_y_pred_ols = []
    all_y_pred_huber = []

    for day in range(1, days_in_month + 1):
        selected_day = f'{year}-{month:02d}-{day:02d}'
        try:
            day_df, _, _ = load_data(selected_day)
            ols_metrics, huber_metrics = run_rrk_ten_fold(day_df)

            # 只收集有效的折数据
            # 运行十折
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

                kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
                          RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
                          WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

                poly = PolynomialFeatures(degree=2, include_bias=False)
                m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
                m_test_poly = poly.transform(m_test.reshape(-1, 1))

                ols = LinearRegression()
                ols.fit(m_train_poly, y_train)
                pred_ols = ols.predict(m_test_poly)
                residual_ols = y_train - ols.predict(m_train_poly)

                huber = HuberRegressor(epsilon=1.35, max_iter=1000)
                huber.fit(m_train_poly, y_train)
                pred_huber = huber.predict(m_test_poly)
                residual_huber = y_train - huber.predict(m_train_poly)

                gpr_ols = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
                gpr_ols.fit(X_train, residual_ols)
                gpr_ols_pred, _ = gpr_ols.predict(X_test, return_std=True)

                gpr_huber = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
                gpr_huber.fit(X_train, residual_huber)
                gpr_huber_pred, _ = gpr_huber.predict(X_test, return_std=True)

                rk_ols_pred = pred_ols + gpr_ols_pred
                rk_huber_pred = pred_huber + gpr_huber_pred

                all_y_true.extend(y_test)
                all_y_pred_ols.extend(rk_ols_pred)
                all_y_pred_huber.extend(rk_huber_pred)

            print(".", end="", flush=True)
        except Exception as e:
            print("x", end="", flush=True)
            continue

    print(" done")

    all_y_true = np.array(all_y_true)
    all_y_pred_ols = np.array(all_y_pred_ols)
    all_y_pred_huber = np.array(all_y_pred_huber)

    ols_metrics = compute_metrics(all_y_true, all_y_pred_ols)
    huber_metrics = compute_metrics(all_y_true, all_y_pred_huber)

    return ols_metrics, huber_metrics


def run_monthly_stage(stage_name, month):
    """运行月度验证阶段"""
    stage_info = STAGES[stage_name]
    print("\n" + "="*70)
    print(f"阶段{stage_name[-1]}：{stage_info['name']}（2020-{month:02d}）")
    print("="*70)
    print(f"判定条件：月平均R² >= {TARGET_R2:.4f}")
    print("-"*70)

    ols_metrics, huber_metrics = run_full_month(month)

    print(f"\n  月度结果:")
    print(f"    OLS:  R²={ols_metrics['R2']:.4f}, RMSE={ols_metrics['RMSE']:.2f}, MAE={ols_metrics['MAE']:.2f}, MB={ols_metrics['MB']:.2f}")
    print(f"    Huber: R²={huber_metrics['R2']:.4f}, RMSE={huber_metrics['RMSE']:.2f}, MAE={huber_metrics['MAE']:.2f}, MB={huber_metrics['MB']:.2f}")

    # 保存月度报告
    rows = [{
        'stage': stage_name,
        'month': f'2020-{month:02d}',
        'method': 'RRK_OLS',
        'R2': ols_metrics['R2'],
        'RMSE': ols_metrics['RMSE'],
        'MAE': ols_metrics['MAE'],
        'MB': ols_metrics['MB'],
    }, {
        'stage': stage_name,
        'month': f'2020-{month:02d}',
        'method': 'RRK_Huber',
        'R2': huber_metrics['R2'],
        'RMSE': huber_metrics['RMSE'],
        'MAE': huber_metrics['MAE'],
        'MB': huber_metrics['MB'],
    }]
    df = pd.DataFrame(rows)
    csv_path = f'{output_dir}/月验证_2020{month:02d}_报告.csv'
    df.to_csv(csv_path, index=False)

    report = f"""# 月验证报告 {stage_info['name']}（2020-{month:02d}）

## 基准
- eVNA: R²={BENCHMARK_R2:.4f}, RMSE={BENCHMARK_RMSE:.2f}, |MB|={BENCHMARK_MB_ABS:.2f}
- 目标: R² >= {TARGET_R2:.4f}

## 结果

| 方法 | R² | RMSE | MAE | MB | R²达标 | RMSE达标 | |MB|达标 |
|------|-----|------|-----|-----|--------|---------|---------|
| RRK-OLS | {ols_metrics['R2']:.4f} | {ols_metrics['RMSE']:.2f} | {ols_metrics['MAE']:.2f} | {ols_metrics['MB']:.2f} | {'是' if ols_metrics['R2'] >= TARGET_R2 else '否'} | {'是' if ols_metrics['RMSE'] <= BENCHMARK_RMSE else '否'} | {'是' if abs(ols_metrics['MB']) <= BENCHMARK_MB_ABS else '否'} |
| RRK-Huber | {huber_metrics['R2']:.4f} | {huber_metrics['RMSE']:.2f} | {huber_metrics['MAE']:.2f} | {huber_metrics['MB']:.2f} | {'是' if huber_metrics['R2'] >= TARGET_R2 else '否'} | {'是' if huber_metrics['RMSE'] <= BENCHMARK_RMSE else '否'} | {'是' if abs(huber_metrics['MB']) <= BENCHMARK_MB_ABS else '否'} |

## 判定
- RRK-OLS: R²={'达标' if ols_metrics['R2'] >= TARGET_R2 else '未达标'}
- RRK-Huber: R²={'达标' if huber_metrics['R2'] >= TARGET_R2 else '未达标'}
"""

    report_path = f'{output_dir}/月验证_2020{month:02d}_报告.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n  报告已保存: {report_path}")
    print(f"  CSV已保存: {csv_path}")

    return {
        'stage': stage_name,
        'month': month,
        'ols': ols_metrics,
        'huber': huber_metrics
    }


def main():
    """主函数"""
    print("="*70)
    print("RRK 多阶段创新验证")
    print("="*70)
    print(f"基准eVNA: R²={BENCHMARK_R2:.4f}, RMSE={BENCHMARK_RMSE:.2f}, |MB|={BENCHMARK_MB_ABS:.2f}")
    print(f"创新目标: R²>={TARGET_R2:.4f}, RMSE<={BENCHMARK_RMSE:.2f}, |MB|<={BENCHMARK_MB_ABS:.2f}")
    print("="*70)

    all_results = {}

    # 阶段1: 预实验5天
    stage1_result = run_stage1_pre_experiment()
    all_results['stage1'] = stage1_result

    if not (stage1_result['pass_ols'] or stage1_result['pass_huber']):
        print("\n" + "="*70)
        print("阶段1未通过，终止验证流程")
        print("="*70)
        return all_results

    print("\n" + "="*70)
    print("阶段1已通过，继续验证")
    print("="*70)

    # 阶段2: 1月
    stage2_result = run_monthly_stage('stage2', 1)
    all_results['stage2'] = stage2_result

    if not (stage2_result['ols']['R2'] >= TARGET_R2 or stage2_result['huber']['R2'] >= TARGET_R2):
        print("\n" + "="*70)
        print("阶段2未通过，终止验证流程")
        print("="*70)
        return all_results

    # 阶段3: 7月
    stage3_result = run_monthly_stage('stage3', 7)
    all_results['stage3'] = stage3_result

    if not (stage3_result['ols']['R2'] >= TARGET_R2 or stage3_result['huber']['R2'] >= TARGET_R2):
        print("\n" + "="*70)
        print("阶段3未通过，终止验证流程")
        print("="*70)
        return all_results

    # 阶段4: 12月
    stage4_result = run_monthly_stage('stage4', 12)
    all_results['stage4'] = stage4_result

    # 生成最终报告
    generate_final_report(all_results)

    return all_results


def generate_final_report(all_results):
    """生成最终汇总报告"""
    print("\n" + "="*70)
    print("生成最终汇总报告")
    print("="*70)

    # 判断最优方法
    methods_status = {}

    for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
        if stage_name not in all_results:
            continue
        r = all_results[stage_name]
        for method in ['ols', 'huber']:
            m = r[method]
            key = f'RRK-{method.upper()}'
            if key not in methods_status:
                methods_status[key] = {'stages': [], 'r2': [], 'rmse': [], 'mb': []}
            methods_status[key]['stages'].append(stage_name)
            methods_status[key]['r2'].append(m['R2'])
            methods_status[key]['rmse'].append(m['RMSE'])
            methods_status[key]['mb'].append(abs(m['MB']))

    # 检查是否所有阶段都通过
    final_status = {}
    for method, status in methods_status.items():
        r2_pass = all(s >= TARGET_R2 for s in status['r2'])
        rmse_pass = all(r <= BENCHMARK_RMSE for r in status['rmse'])
        mb_pass = all(m <= BENCHMARK_MB_ABS for m in status['mb'])
        final_status[method] = {
            'r2_pass': r2_pass,
            'rmse_pass': rmse_pass,
            'mb_pass': mb_pass,
            'avg_r2': np.mean(status['r2']),
            'avg_rmse': np.mean(status['rmse']),
            'avg_mb': np.mean(status['mb']),
            'all_pass': r2_pass and rmse_pass and mb_pass
        }

    # 更新innovation_summary.csv
    summary_rows = []
    for method, status in final_status.items():
        innovation_status = '创新成立' if status['all_pass'] else '创新不成立'
        summary_rows.append({
            'method': f'RK_Huber_Poly' if 'HUBER' in method else 'RK_OLS_Poly',
            'R2': status['avg_r2'],
            'RMSE': status['avg_rmse'],
            'MAE': np.nan,  # 会在下面更新
            'MB': status['avg_mb'],
            'stage1_avg': all_results.get('stage1', {}).get('huber' if 'HUBER' in method else 'ols', {}).get('R2', np.nan),
            'stage2_avg': all_results.get('stage2', {}).get('huber' if 'HUBER' in method else 'ols', {}).get('R2', np.nan),
            'stage3_avg': all_results.get('stage3', {}).get('huber' if 'HUBER' in method else 'ols', {}).get('R2', np.nan),
            'stage4_avg': all_results.get('stage4', {}).get('huber' if 'HUBER' in method else 'ols', {}).get('R2', np.nan),
            'status': innovation_status
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = f'{root_dir}/test_result/innovation_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"汇总CSV已更新: {summary_csv}")

    # 更新快照
    try:
        sys.path.insert(0, f'{root_dir}/test_result')
        from snapshot_manager import SnapshotManager
        manager = SnapshotManager(root_dir)

        for method, status in final_status.items():
            if status['all_pass']:
                method_name = f'RK_Huber_Poly' if 'HUBER' in method else 'RK_OLS_Poly'
                manager.update_best_method(method_name, {'R2': status['avg_r2'], 'RMSE': status['avg_rmse'], 'MB': status['avg_mb']})
                print(f"快照已更新: {method_name}")
    except Exception as e:
        print(f"快照更新失败: {e}")

    # 打印最终结果
    print("\n" + "="*70)
    print("最终验证结果")
    print("="*70)
    print(f"{'方法':<20} {'R²':>10} {'RMSE':>10} {'|MB|':>10} {'状态':<15}")
    print("-"*65)
    for method, status in final_status.items():
        print(f"{method:<20} {status['avg_r2']:>10.4f} {status['avg_rmse']:>10.2f} {status['avg_mb']:>10.2f} {'创新成立' if status['all_pass'] else '创新不成立'}")

    return final_status


if __name__ == '__main__':
    results = main()