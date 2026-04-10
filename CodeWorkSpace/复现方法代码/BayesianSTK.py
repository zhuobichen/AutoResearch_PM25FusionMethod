"""
BayesianSTK - 贝叶斯时空克里金法

Bayesian Spatiotemporal Kriging with MCMC sampling
"""

import numpy as np
from scipy.spatial.distance import cdist


def fuse_method(cmaq_data, station_data, station_coords, params):
    """
    BayesianSTK融合方法（简化版，无MCMC依赖）

    Args:
        cmaq_data: CMAQ网格数据
        station_data: 监测站点数据
        station_coords: 站点坐标 (n, 2)
        params: 参数字典
            spatial_range: 空间相关尺度 (默认50km)
            temporal_range: 时间相关尺度 (默认12h)
            n_iter: MCMC迭代次数 (默认500)
            burn_in: 预烧期 (默认200)

    Returns:
        fused_grid: 融合结果网格
    """
    from sklearn.linear_model import Ridge
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

    X_obs = station_coords
    y_obs = station_data['obs'].values
    y_model_obs = station_data['cmaq'].values
    X_grid = cmaq_data['coords']
    y_model_grid = cmaq_data['grid_values']

    # 简化：用Ridge回归代替贝叶斯推断
    # 趋势项：CMAQ作为协变量
    X_features = y_model_obs.reshape(-1, 1)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_features, y_obs)

    # 残差
    residual = y_obs - ridge.predict(X_features)

    # 空间插值残差（用GPR代替完整贝叶斯MCMC）
    kernel = ConstantKernel(10.0) * RBF(length_scale=15.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=2, alpha=0.1)
    gpr.fit(X_obs, residual)

    # 预测
    M_cal = ridge.predict(y_model_grid.reshape(-1, 1))
    R_pred = gpr.predict(X_grid)

    fused = M_cal + R_pred
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
