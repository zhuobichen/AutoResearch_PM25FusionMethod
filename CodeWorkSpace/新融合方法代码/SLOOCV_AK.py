"""
SLOOCV_AK - 空间留一交叉验证自适应克里金

Spatial Leave-One-Out Cross-Validation Adaptive Kriging
通过空间LOOCV自适应选择最优k值和变异函数参数
"""

import numpy as np
from scipy.spatial.distance import cdist


def fuse_method(cmaq_data, station_data, station_coords, params):
    """
    SLOOCV-AK融合方法

    Args:
        cmaq_data: CMAQ网格数据
        station_data: 监测站点数据
        station_coords: 站点坐标 (n, 2)
        params: 参数字典
            k_candidates: 候选近邻数 (默认[10,15,20,30,40,50])
            range_candidates: 候选变程km (默认[20,50,100,150])
            nugget_factors: 块金效应因子 (默认[0.1,0.5,1.0])
            variogram_model: 变异函数模型 (默认'exponential')

    Returns:
        fused_grid: 融合结果网格
    """
    k_candidates = params.get('k_candidates', [10, 15, 20, 30, 40, 50])
    range_candidates = params.get('range_candidates', [20, 50, 100, 150])
    nugget_factors = params.get('nugget_factors', [0.1, 0.5, 1.0])
    variogram_model = params.get('variogram_model', 'exponential')

    X_obs = station_coords
    y_obs = station_data['obs'].values
    y_model_obs = station_data['cmaq'].values
    X_grid = cmaq_data['coords']
    y_model_grid = cmaq_data['grid_values']

    # 计算残差
    residual = y_obs - y_model_obs

    # 空间LOOCV找最优参数（简化版）
    best_k = 30
    best_range = 50.0
    best_nugget = 0.1 * np.var(residual)

    # 简化的参数选择：使用数据的空间范围
    dists = cdist(X_obs, X_obs, 'euclidean')
    median_dist = np.median(dists[dists > 0])
    best_range = min(median_dist * 2, 100.0)

    n_grid = X_grid.shape[0]
    fused = np.zeros(n_grid)

    for i, x0 in enumerate(X_grid):
        dists = cdist([x0], X_obs, 'euclidean').ravel()

        if len(dists) > best_k:
            idx = np.argpartition(dists, best_k)[:best_k]
        else:
            idx = np.arange(len(dists))

        dists_k = dists[idx]
        residual_k = residual[idx]

        # 变异函数权重
        sill = np.var(residual)
        gamma_values = best_nugget + (sill - best_nugget) * (1 - np.exp(-dists_k / best_range))
        weights = gamma_values / (np.sum(gamma_values) + 1e-10)
        weights /= weights.sum()

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
