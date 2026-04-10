"""
VNA - Voronoi Neighbor Averaging (冯洛诺伊邻域平均法)

公式: P_VNA(s_0) = sum(w_i * O(s_i))
其中 w_i = d_i^p / sum(d_j^p), p = -2 (反距离加权)
"""

import numpy as np
from scipy.spatial.distance import cdist


def fuse_method(cmaq_data, station_data, station_coords, params):
    """
    VNA融合方法

    Args:
        cmaq_data: CMAQ网格数据
        station_data: 监测站点数据
        station_coords: 站点坐标 (n, 2) - [lon, lat]
        params: 参数字典
            k: 近邻数量 (默认30)
            power: 距离权重指数 (默认-2，表示1/d^2)
            min_neighbors: 最小近邻数 (默认3)

    Returns:
        fused_grid: 融合结果网格
    """
    k = params.get('k', 30)
    power = params.get('power', -2)
    min_neighbors = params.get('min_neighbors', 3)

    X_obs = station_coords
    y_obs = station_data['obs'].values
    X_grid = cmaq_data['coords']  # 网格点坐标
    y_model_grid = cmaq_data['grid_values']  # 网格CMAQ值

    n_grid = X_grid.shape[0]
    fused = np.zeros(n_grid)

    for i, x0 in enumerate(X_grid):
        # 计算到所有站点的距离
        dists = cdist([x0], X_obs, 'euclidean').ravel()

        # 找到k个最近邻
        if len(dists) > k:
            idx = np.argpartition(dists, k)[:k]
        else:
            idx = np.arange(len(dists))

        # 检查最小近邻数
        if len(idx) < min_neighbors:
            fused[i] = np.mean(y_obs)
            continue

        dists_k = dists[idx]
        dists_k = np.maximum(dists_k, 1e-6)

        # IDW权重: w_i = 1/d_i^2
        if power < 0:
            weights = dists_k ** (-power)
        else:
            weights = 1.0 / (dists_k ** power)
        weights /= weights.sum()

        # 加权平均
        fused[i] = np.sum(weights * y_obs[idx])

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
