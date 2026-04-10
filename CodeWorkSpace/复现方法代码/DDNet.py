"""
DDNet - 双深度神经网络法

Dual Deep Neural Networks for PM2.5 Forecasting and Data Assimilation
PredNet预测网络 + DANet数据同化网络
"""

import numpy as np


def fuse_method(cmaq_data, station_data, station_coords, params):
    """
    DDNet融合方法（简化版，无神经网络依赖）

    注意：完整DDNet需要PyTorch训练，这里提供简化版本
    使用线性回归模拟PredNet-DANet偏差校正

    Args:
        cmaq_data: CMAQ网格数据
        station_data: 监测站点数据
        station_coords: 站点坐标 (n, 2)
        params: 参数字典
            hidden_dim: 隐变量维度 (默认128)
            learning_rate: 学习率 (默认1e-3)

    Returns:
        fused_grid: 融合结果网格
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X_obs = station_coords
    y_obs = station_data['obs'].values
    y_model_obs = station_data['cmaq'].values
    X_grid = cmaq_data['coords']
    y_model_grid = cmaq_data['grid_values']

    # 简化：使用Ridge回归模拟DANet的偏差学习
    # 特征：站点坐标 + CMAQ值
    scaler = StandardScaler()
    X_features = scaler.fit_transform(np.column_stack([X_obs, y_model_obs]))

    # 训练偏差模型
    bias = y_obs - y_model_obs
    model = Ridge(alpha=1.0)
    model.fit(X_features, bias)

    # 预测网格偏差
    X_grid_features = scaler.transform(np.column_stack([X_grid, y_model_grid]))
    grid_bias_pred = model.predict(X_grid_features)

    fused = y_model_grid + grid_bias_pred
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
