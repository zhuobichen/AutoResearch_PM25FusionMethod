"""
DiffusionSmoothing - 扩散平滑法

基于热方程的扩散平滑方法
公式: 模拟热扩散过程平滑CMAQ偏差场
"""

import numpy as np
from scipy.spatial.distance import cdist


def fuse_method(cmaq_data, station_data, station_coords, params):
    """
    DiffusionSmoothing融合方法

    Args:
        cmaq_data: CMAQ网格数据
        station_data: 监测站点数据
        station_coords: 站点坐标 (n, 2) - [lon, lat]
        params: 参数字典
            diffusion_coeff: 扩散系数 (默认1.0)
            n_iterations: 迭代次数 (默认50)
            k: 近邻数量 (默认30)

    Returns:
        fused_grid: 融合结果网格
    """
    diffusion_coeff = params.get('diffusion_coeff', 1.0)
    n_iterations = params.get('n_iterations', 50)
    k = params.get('k', 30)

    X_obs = station_coords
    y_obs = station_data['obs'].values
    y_model_obs = station_data['cmaq'].values
    X_grid = cmaq_data['coords']
    y_model_grid = cmaq_data['grid_values']

    # 计算站点偏差
    bias = y_obs - y_model_obs

    # 初始化网格偏差场（用IDW插值）
    n_grid = X_grid.shape[0]
    grid_bias = np.zeros(n_grid)

    for i, x0 in enumerate(X_grid):
        dists = cdist([x0], X_obs, 'euclidean').ravel()
        if len(dists) > k:
            idx = np.argpartition(dists, k)[:k]
        else:
            idx = np.arange(len(dists))

        dists_k = dists[idx]
        dists_k = np.maximum(dists_k, 1e-6)
        weights = 1.0 / (dists_k ** 2)
        weights /= weights.sum()
        grid_bias[i] = np.sum(weights * bias[idx])

    # 扩散平滑迭代
    for iteration in range(n_iterations):
        new_grid_bias = grid_bias.copy()

        for i, x0 in enumerate(X_grid):
            dists = cdist([x0], X_grid, 'euclidean').ravel()

            if len(dists) > k:
                idx = np.argpartition(dists, k)[:k]
            else:
                idx = np.arange(len(dists))

            dists_k = dists[idx]
            dists_k = np.maximum(dists_k, 1e-6)

            # 高斯核权重
            weights = np.exp(-dists_k ** 2 / (4 * diffusion_coeff))
            weights /= weights.sum()

            # 拉普拉斯算子近似
            laplacian = np.sum(weights * (grid_bias[idx] - grid_bias[i]))
            new_grid_bias[i] = grid_bias[i] + diffusion_coeff * laplacian

        grid_bias = new_grid_bias

    # 融合：M + 平滑后的偏差
    fused = y_model_grid + grid_bias

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
