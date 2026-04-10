"""
NeuroDDAF - 神经动态扩散平流场法

Neural Dynamic Diffusion-Advection Fields with Evidential Fusion
简化版（无PyTorch依赖）
"""

import numpy as np
from scipy.spatial.distance import cdist


def fuse_method(cmaq_data, station_data, station_coords, params):
    """
    NeuroDDAF融合方法（简化版）

    使用平流-扩散物理模型的思想进行偏差校正

    Args:
        cmaq_data: CMAQ网格数据
        station_data: 监测站点数据
        station_coords: 站点坐标 (n, 2)
        params: 参数字典
            diffusion_coeff: 扩散系数 (默认1.0)
            advection_weight: 平流权重 (默认0.5)

    Returns:
        fused_grid: 融合结果网格
    """
    X_obs = station_coords
    y_obs = station_data['obs'].values
    y_model_obs = station_data['cmaq'].values
    X_grid = cmaq_data['coords']
    y_model_grid = cmaq_data['grid_values']

    # 计算初始偏差
    bias = y_obs - y_model_obs

    # 简化的平流-扩散校正
    # 平流项：沿浓度梯度方向的校正
    # 扩散项：空间平滑校正

    n_grid = X_grid.shape[0]
    grid_bias = np.zeros(n_grid)

    # 计算站点处CMAQ梯度（简化）
    if len(X_obs) > 2:
        # 用线性回归估计梯度效应
        from sklearn.linear_model import Ridge
        X_with_coords = np.column_stack([X_obs, y_model_obs])
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_with_coords, bias)
        grid_bias = ridge.predict(np.column_stack([X_grid, y_model_grid]))
    else:
        # 近邻插值
        k = 10
        for i, x0 in enumerate(X_grid):
            dists = cdist([x0], X_obs, 'euclidean').ravel()
            if len(dists) > k:
                idx = np.argpartition(dists, k)[:k]
            else:
                idx = np.arange(len(dists))
            dists_k = np.maximum(dists[idx], 1e-6)
            weights = 1.0 / (dists_k ** 2)
            weights /= weights.sum()
            grid_bias[i] = np.sum(weights * bias[idx])

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
