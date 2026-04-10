"""
VG_VNA - 变异函数几何加权VNA

Variogram-Geometric VNA
用变异函数建模的空间相关性替代简单反距离权重
"""

import numpy as np
from scipy.spatial.distance import cdist


def fuse_method(cmaq_data, station_data, station_coords, params):
    """
    VG-VNA融合方法

    Args:
        cmaq_data: CMAQ网格数据
        station_data: 监测站点数据
        station_coords: 站点坐标 (n, 2)
        params: 参数字典
            variogram_model: 变异函数模型 (默认'exponential')
            n_lags: 距离滞后数量 (默认15)
            lag_size: 滞后间距km (默认5.0)
            max_range: 最大变程km (默认200.0)
            k: 近邻数量 (默认30)

    Returns:
        fused_grid: 融合结果网格
    """
    variogram_model = params.get('variogram_model', 'exponential')
    n_lags = params.get('n_lags', 15)
    lag_size = params.get('lag_size', 5.0)
    max_range = params.get('max_range', 200.0)
    k = params.get('k', 30)

    X_obs = station_coords
    y_obs = station_data['obs'].values
    y_model_obs = station_data['cmaq'].values
    X_grid = cmaq_data['coords']
    y_model_grid = cmaq_data['grid_values']

    # 计算残差
    residual = y_obs - y_model_obs

    # 估计变异函数参数
    sill = np.var(residual)
    range_param = min(max_range, np.median(cdist(X_obs, X_obs).ravel()) * 2)
    nugget = 0.0

    n_grid = X_grid.shape[0]
    fused = np.zeros(n_grid)

    for i, x0 in enumerate(X_grid):
        dists = cdist([x0], X_obs, 'euclidean').ravel()

        if len(dists) > k:
            idx = np.argpartition(dists, k)[:k]
        else:
            idx = np.arange(len(dists))

        dists_k = dists[idx]
        residual_k = residual[idx]

        # 用变异函数计算权重
        # gamma(h) = nugget + sill * (1 - exp(-h/range))
        gamma_values = nugget + sill * (1 - np.exp(-dists_k / range_param))

        # 克里金权重：w = gamma / sum(gamma)
        weights = gamma_values / (np.sum(gamma_values) + 1e-10)
        weights /= weights.sum()

        # 预测残差
        residual_pred = np.sum(weights * residual_k)
        fused[i] = y_model_grid[i] + residual_pred

    return fused


def calculate_metrics(y_true, y_pred):
    """计算R2、MAE、RMSE、MB"""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return {
        'R2': r2_score(y_true[mask], y_pred[mask]),
        'MAE': mean_absolute_error(y_true[mask], y_pred[mask]),
        'RMSE': np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
        'MB': np.mean(y_pred[mask] - y_true[mask])
    }
