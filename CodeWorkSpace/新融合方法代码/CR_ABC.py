"""
CR_ABC - 浓度分层自适应偏差校正

Concentration-Regime Adaptive Bias Correction
PM2.5偏差呈现非线性特征：模型在低浓度时倾向于高估，高浓度时倾向于低估
"""

import numpy as np
from scipy.spatial.distance import cdist


def fuse_method(cmaq_data, station_data, station_coords, params):
    """
    CR-ABC融合方法

    Args:
        cmaq_data: CMAQ网格数据
        station_data: 监测站点数据
        station_coords: 站点坐标 (n, 2)
        params: 参数字典
            n_levels: 浓度分层数 (默认3)
            percentile_low: 低层分位数阈值 (默认33.0)
            percentile_high: 高层分位数阈值 (默认67.0)
            transition_width: sigmoid过渡宽度 (默认0.25)
            k: 近邻数量 (默认30)
            power: 距离权重指数 (默认-2)

    Returns:
        fused_grid: 融合结果网格
    """
    n_levels = params.get('n_levels', 3)
    percentile_low = params.get('percentile_low', 33.0)
    percentile_high = params.get('percentile_high', 67.0)
    transition_width = params.get('transition_width', 0.25)
    k = params.get('k', 30)
    power = params.get('power', -2)

    X_obs = station_coords
    y_obs = station_data['obs'].values
    y_model_obs = station_data['cmaq'].values
    X_grid = cmaq_data['coords']
    y_model_grid = cmaq_data['grid_values']

    # 计算偏差
    bias = y_obs - y_model_obs

    # 计算分位数阈值
    Q33 = np.percentile(y_model_obs, percentile_low)
    Q67 = np.percentile(y_model_obs, percentile_high)
    Q50 = np.percentile(y_model_obs, 50.0)

    # 对站点进行分层
    layers = np.zeros(len(y_model_obs), dtype=int)
    layers[y_model_obs < Q33] = 0  # low
    layers[(y_model_obs >= Q33) & (y_model_obs < Q67)] = 1  # medium
    layers[y_model_obs >= Q67] = 2  # high

    # 计算每层的加权偏差
    layer_bias = {}
    for layer_id in range(3):
        mask = layers == layer_id
        if np.sum(mask) < 2:
            layer_bias[layer_id] = np.mean(bias)
        else:
            layer_bias[layer_id] = np.mean(bias[mask])

    # sigmoid过渡宽度
    sigma = (Q67 - Q33) / 4

    n_grid = X_grid.shape[0]
    fused = np.zeros(n_grid)

    for i, x0 in enumerate(X_grid):
        M_val = y_model_grid[i]

        # 计算sigmoid权重
        if sigma > 0:
            alpha = 1.0 / (1.0 + np.exp(-(M_val - Q50) / sigma))
        else:
            alpha = 0.5

        # 根据浓度值选择对应层的偏差
        if M_val < Q33:
            bias_pred = layer_bias[0]
        elif M_val >= Q67:
            bias_pred = layer_bias[2]
        else:
            # 使用sigmoid插值
            bias_pred = (1 - alpha) * layer_bias[0] + alpha * layer_bias[2]

        fused[i] = M_val + bias_pred

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
