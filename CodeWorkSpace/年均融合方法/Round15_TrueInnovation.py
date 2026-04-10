"""
年平均数据 - 第十五轮：真正的单模型创新
==========================================

当前最佳: SpatialStratifiedFusion R²=0.5787
目标: R²≥0.5800 (差距0.0013)

核心创新:
1. ResidualKrigingFusion: 使用克里金进行残差空间插值
2. HybridTrendKrigingFusion: 结合非线性趋势和克里金
3. CMAQTransformedFusion: CMAQ非线性变换后融合
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch')

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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


class ResidualKrigingFusion:
    """
    ResidualKrigingFusion: 残差克里金融合

    原理:
    1. 使用CMAQ作为趋势项（线性回归）
    2. 计算观测与趋势的残差
    3. 使用克里金插值残差
    4. 最终预测 = 趋势 + 克里金残差
    """
    def __init__(self, k=20, variogram_model='exponential'):
        self.k = k
        self.variogram_model = variogram_model

    def fit(self, coords, y_obs, y_model):
        """拟合模型"""
        self.coords = coords
        self.y_obs = y_obs
        self.y_model = y_model
        self.n = len(y_obs)

        # 1. 趋势建模
        X = np.column_stack([np.ones(self.n), y_model])
        XT_X = X.T @ X
        XT_y = X.T @ y_obs
        self.beta = np.linalg.solve(XT_X + 1e-6 * np.eye(2), XT_y)
        self.trend = X @ self.beta

        # 2. 残差
        self.residuals = y_obs - self.trend

        # 3. 拟合变异函数
        self._fit_variogram()

    def _fit_variogram(self):
        """拟合半变异函数"""
        dists = cdist(self.coords, self.coords, 'euclidean')
        idx_i, idx_j = np.triu_indices(self.n)
        h = dists[idx_i, idx_j]
        gamma_exp = 0.5 * (self.residuals[idx_i] - self.residuals[idx_j]) ** 2

        mask = h > 1e-6
        h = h[mask]
        gamma_exp = gamma_exp[mask]

        self.nugget = 0
        self.sill = np.var(self.residuals)
        self.range_param = np.median(h) if len(h) > 0 and np.median(h) > 0.1 else 1.0

    def _variogram_func(self, h):
        """变异函数模型"""
        c0 = self.nugget
        c = self.sill - self.nugget
        a = max(self.range_param, 0.01)

        if self.variogram_model == 'exponential':
            return c0 + c * (1 - np.exp(-h / a))
        elif self.variogram_model == 'spherical':
            result = np.zeros_like(h, dtype=float)
            mask = h <= a
            result[mask] = c0 + c * (1.5 * h[mask] / a - 0.5 * (h[mask] / a) ** 3)
            result[~mask] = c0 + c
            return result
        elif self.variogram_model == 'gaussian':
            return c0 + c * (1 - np.exp(-(h / a) ** 2))
        else:
            return c0 + c * (1 - np.exp(-h / a))

    def predict(self, X_grid, y_model_grid):
        """预测"""
        n_grid = X_grid.shape[0]

        # 趋势项
        X_trend = np.column_stack([np.ones(n_grid), y_model_grid])
        trend_grid = X_trend @ self.beta

        # 克里金残差
        residual_pred = np.zeros(n_grid)

        # 计算观测点间距离矩阵
        dists_obs = cdist(self.coords, self.coords, 'euclidean')
        Gamma = self._variogram_func(dists_obs)

        # 通用克里金系统矩阵
        p = 2
        A = np.zeros((self.n + p, self.n + p))
        A[:self.n, :self.n] = Gamma
        A[:self.n, self.n:] = np.column_stack([np.ones(self.n), self.y_model])
        A[self.n:, :self.n] = np.column_stack([np.ones(self.n), self.y_model]).T

        for i in range(n_grid):
            dists_to_obs = cdist([X_grid[i]], self.coords, 'euclidean').ravel()
            gamma_0 = self._variogram_func(dists_to_obs)
            x_0 = np.array([1, y_model_grid[i]])

            b = np.zeros(self.n + p)
            b[:self.n] = gamma_0
            b[self.n:] = x_0

            try:
                weights = np.linalg.solve(A + 1e-6 * np.eye(self.n + p), b)
                lambda_i = weights[:self.n]
            except:
                idx = np.argpartition(dists_to_obs, min(self.k, self.n-1))[:min(self.k, self.n-1)]
                lambda_i = np.zeros(self.n)
                lambda_i[idx] = 1.0 / (dists_to_obs[idx] + 1e-6)
                lambda_i /= lambda_i.sum()

            residual_pred[i] = np.sum(lambda_i * self.residuals)

        y_pred = trend_grid + residual_pred
        return np.maximum(y_pred, 0)


def run_residual_kriging_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """ResidualKrigingFusion - 克里金残差融合"""
    print("\n=== ResidualKrigingFusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        coords_train = train_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values
        m_train = train_df['CMAQ'].values

        coords_test = test_df[['Lon', 'Lat']].values
        m_test = test_df['CMAQ'].values

        # ResidualKrigingFusion
        rkf = ResidualKrigingFusion(k=20, variogram_model='exponential')
        rkf.fit(coords_train, y_train, m_train)
        rkf_pred_grid = rkf.predict(X_grid_full, y_grid_model_full)

        # 提取预测
        rkf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dist = np.sqrt((lon_cmaq - coords_test[i, 0])**2 + (lat_cmaq - coords_test[i, 1])**2)
            idx = np.argmin(dist)
            row_idx, col_idx = idx // nx, idx % nx
            rkf_pred[i] = rkf_pred_grid[row_idx * nx + col_idx]

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'rkf': rkf_pred
        }

        print(f"  Fold {fold_id}: completed")

    rkf_all = np.concatenate([results[f]['rkf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, rkf_all)
    print(f"  ResidualKrigingFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_hybrid_trend_kriging_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    HybridTrendKrigingFusion: 混合趋势克里金融合

    结合:
    - CMAQ线性趋势
    - NNA偏差校正
    - 克里金残差插值
    """
    print("\n=== HybridTrendKrigingFusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        coords_train = train_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values
        m_train = train_df['CMAQ'].values

        coords_test = test_df[['Lon', 'Lat']].values
        m_test = test_df['CMAQ'].values

        # 1. 克里金趋势
        rkf = ResidualKrigingFusion(k=20, variogram_model='exponential')
        rkf.fit(coords_train, y_train, m_train)
        trend_pred_grid = rkf.predict(X_grid_full, y_grid_model_full)

        # 2. NNA偏差校正
        bias = y_train - m_train
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(coords_train, bias)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_grid = y_grid_model_full + bias_grid

        # 3. NNA ratio校正
        ratio = y_train / m_train
        nn_ratio = NNA(method='nearest', k=15, power=-2)
        nn_ratio.fit(coords_train, ratio)
        ratio_grid = nn_ratio.predict(X_grid_full)
        evna_grid = y_grid_model_full * ratio_grid

        obs_mean = y_train.mean()
        obs_std = y_train.std()

        htkf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            cmaq_dev = np.abs(m_test[i] - obs_mean) / obs_std

            # 基于CMAQ偏差的权重
            prior_trend = np.clip(0.50 - 0.25 * cmaq_dev, 0.25, 0.55)

            # 计算训练集残差标准差
            rkf_res = y_train - rkf.predict(coords_train, m_train)
            avna_res = y_train - (m_train + nn_bias.predict(coords_train))
            evna_res = y_train - (m_train * nn_ratio.predict(coords_train))

            std_rkf = np.std(rkf_res) + 1e-6
            std_avna = np.std(avna_res) + 1e-6
            std_evna = np.std(evna_res) + 1e-6

            # 似然权重
            like_rkf = 1.0 / std_rkf
            like_avna = 1.0 / std_avna
            like_evna = 1.0 / std_evna

            w_trend = prior_trend
            w_avna = (1 - prior_trend) * like_avna / (like_avna + like_evna)
            w_evna = (1 - prior_trend) * like_evna / (like_avna + like_evna)

            total = w_trend + w_avna + w_evna
            w_trend /= total
            w_avna /= total
            w_evna /= total

            # 提取预测
            dist = np.sqrt((lon_cmaq - coords_test[i, 0])**2 + (lat_cmaq - coords_test[i, 1])**2)
            idx = np.argmin(dist)
            row_idx, col_idx = idx // nx, idx % nx
            trend_i = trend_pred_grid[row_idx * nx + col_idx]
            avna_i = avna_grid[row_idx * nx + col_idx]
            evna_i = evna_grid[row_idx * nx + col_idx]

            htkf_pred[i] = w_trend * trend_i + w_avna * avna_i + w_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'htkf': htkf_pred
        }

        print(f"  Fold {fold_id}: completed")

    htkf_all = np.concatenate([results[f]['htkf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, htkf_all)
    print(f"  HybridTrendKrigingFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def run_cmaq_transformed_fusion(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg):
    """
    CMAQTransformedFusion: CMAQ变换融合

    原理:
    - 对CMAQ进行对数变换使残差分布更正常
    - 结合VNA使用CMAQ变换后的偏差
    """
    print("\n=== CMAQTransformedFusion ===")

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_year_avg.ravel()

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        if len(test_df) == 0:
            continue

        coords_train = train_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values
        m_train = train_df['CMAQ'].values

        coords_test = test_df[['Lon', 'Lat']].values
        m_test = test_df['CMAQ'].values

        # 1. VNA
        nn_vna = NNA(method='nearest', k=15, power=-2)
        nn_vna.fit(coords_train, y_train)
        vna_grid = nn_vna.predict(X_grid_full)

        # 2. Log-CMAQ偏差校正
        log_m = np.log(m_train + 1)
        log_bias = y_train - log_m
        nn_log_bias = NNA(method='nearest', k=15, power=-2)
        nn_log_bias.fit(coords_train, log_bias)
        log_bias_grid = nn_log_bias.predict(X_grid_full)
        log_avna_grid = np.exp(log_m.mean() + log_bias_grid)

        # 3. 标准CMAQ偏差校正
        bias = y_train - m_train
        nn_bias = NNA(method='nearest', k=15, power=-2)
        nn_bias.fit(coords_train, bias)
        bias_grid = nn_bias.predict(X_grid_full)
        avna_grid = y_grid_model_full + bias_grid

        # 4. eVNA
        ratio = y_train / m_train
        nn_ratio = NNA(method='nearest', k=15, power=-2)
        nn_ratio.fit(coords_train, ratio)
        ratio_grid = nn_ratio.predict(X_grid_full)
        evna_grid = y_grid_model_full * ratio_grid

        obs_mean = y_train.mean()
        obs_std = y_train.std()

        ctf_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            cmaq_dev = np.abs(m_test[i] - obs_mean) / obs_std

            # 基于CMAQ偏差的权重
            prior_vna = np.clip(0.68 - 0.32 * cmaq_dev, 0.28, 0.68)

            # 似然权重
            log_avna_res = y_train - (np.exp(log_m.mean() + nn_log_bias.predict(coords_train)))
            avna_res = y_train - (m_train + nn_bias.predict(coords_train))
            evna_res = y_train - (m_train * nn_ratio.predict(coords_train))

            std_log_avna = np.std(log_avna_res) + 1e-6
            std_avna = np.std(avna_res) + 1e-6
            std_evna = np.std(evna_res) + 1e-6

            like_log_avna = 1.0 / std_log_avna
            like_avna = 1.0 / std_avna
            like_evna = 1.0 / std_evna

            w_vna = prior_vna
            w_log_avna = (1 - prior_vna) * 0.3 * like_log_avna / (like_log_avna + like_avna + like_evna)
            w_avna = (1 - prior_vna) * 0.4 * like_avna / (like_log_avna + like_avna + like_evna)
            w_evna = (1 - prior_vna) * 0.3 * like_evna / (like_log_avna + like_avna + like_evna)

            total = w_vna + w_log_avna + w_avna + w_evna
            w_vna /= total
            w_log_avna /= total
            w_avna /= total
            w_evna /= total

            # 提取预测
            dist = np.sqrt((lon_cmaq - coords_test[i, 0])**2 + (lat_cmaq - coords_test[i, 1])**2)
            idx = np.argmin(dist)
            row_idx, col_idx = idx // nx, idx % nx
            vna_i = vna_grid[row_idx * nx + col_idx]
            log_avna_i = log_avna_grid[row_idx * nx + col_idx]
            avna_i = avna_grid[row_idx * nx + col_idx]
            evna_i = evna_grid[row_idx * nx + col_idx]

            ctf_pred[i] = w_vna * vna_i + w_log_avna * log_avna_i + w_avna * avna_i + w_evna * evna_i

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'ctf': ctf_pred
        }

        print(f"  Fold {fold_id}: completed")

    ctf_all = np.concatenate([results[f]['ctf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(true_all, ctf_all)
    print(f"  CMAQTransformedFusion: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics, results


def main():
    print("="*60)
    print("第十五轮：真正的单模型创新")
    print("="*60)

    year_df, lon_cmaq, lat_cmaq, cmaq_year_avg = load_and_prepare_year_avg_data()

    results = {}

    methods = [
        ('ResidualKrigingFusion', run_residual_kriging_fusion),
        ('HybridTrendKrigingFusion', run_hybrid_trend_kriging_fusion),
        ('CMAQTransformedFusion', run_cmaq_transformed_fusion),
    ]

    for name, func in methods:
        metrics, _ = func(year_df, lon_cmaq, lat_cmaq, cmaq_year_avg)
        results[name] = metrics

    print("\n" + "="*60)
    print("第十五轮方法结果汇总")
    print("="*60)

    results_df = pd.DataFrame([
        {'method': method, **metrics}
        for method, metrics in results.items()
    ])
    results_df = results_df.sort_values('R2', ascending=False)
    results_df.to_csv(f'{output_dir}/fifteenth_round_results.csv', index=False)

    print("\n方法排名 (按R2降序):")
    for _, row in results_df.iterrows():
        print(f"  {row['method']:30s}: R2={row['R2']:.4f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}, MB={row['MB']:.2f}")

    # 基准对比
    print("\n" + "="*60)
    print("基准对比 (VNA R2=0.5700, 目标 R2>=0.5800)")
    print("="*60)
    for _, row in results_df.iterrows():
        improvement = row['R2'] - 0.5700
        status = "达标" if improvement >= 0.01 else "未达标"
        print(f"  {row['method']:30s}: 提升 {improvement:+.4f} [{status}]")

    # 与之前最佳对比
    print("\n" + "="*60)
    print("与之前最佳对比 (SpatialStratifiedFusion R2=0.5787)")
    print("="*60)
    for _, row in results_df.iterrows():
        diff = row['R2'] - 0.5787
        print(f"  {row['method']:30s}: 差距 {diff:+.4f}")

    return results_df


if __name__ == '__main__':
    results_df = main()