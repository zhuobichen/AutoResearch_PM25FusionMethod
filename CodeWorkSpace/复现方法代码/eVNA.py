"""
eVNA - Enhanced VNA (比率法)

公式: P_eVNA(s_0) = M(s_0) * sum(w_i * O(s_i)/M(s_i))
基于比率法的空间插值
"""

import numpy as np
from scipy.spatial.distance import cdist


def fuse_method(cmaq_data, station_data, station_coords, params):
    """
    eVNA融合方法

    Args:
        cmaq_data: CMAQ网格数据
        station_data: 监测站点数据
        station_coords: 站点坐标 (n, 2) - [lon, lat]
        params: 参数字典
            k: 近邻数量 (默认30)
            power: 距离权重指数 (默认-2)
            min_neighbors: 最小近邻数 (默认3)
            epsilon: 防止除零的小常数 (默认1e-6)

    Returns:
        fused_grid: 融合结果网格
    """
    k = params.get('k', 30)
    power = params.get('power', -2)
    min_neighbors = params.get('min_neighbors', 3)
    epsilon = params.get('epsilon', 1e-6)

    X_obs = station_coords
    y_obs = station_data['obs'].values
    y_model_obs = station_data['cmaq'].values  # 站点CMAQ值
    X_grid = cmaq_data['coords']
    y_model_grid = cmaq_data['grid_values']

    # 计算比率场 O/M
    ratio = y_obs / (y_model_obs + epsilon)

    n_grid = X_grid.shape[0]
    fused = np.zeros(n_grid)

    for i, x0 in enumerate(X_grid):
        dists = cdist([x0], X_obs, 'euclidean').ravel()

        if len(dists) > k:
            idx = np.argpartition(dists, k)[:k]
        else:
            idx = np.arange(len(dists))

        if len(idx) < min_neighbors:
            fused[i] = y_model_grid[i]
            continue

        dists_k = dists[idx]
        dists_k = np.maximum(dists_k, epsilon)

        if power < 0:
            weights = dists_k ** (-power)
        else:
            weights = 1.0 / (dists_k ** power)
        weights /= weights.sum()

        # 插值比率
        ratio_interp = np.sum(weights * ratio[idx])
        fused[i] = y_model_grid[i] * ratio_interp

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
