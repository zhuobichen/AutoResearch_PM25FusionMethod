"""
IDW偏差加权融合法 (Inverse Distance Weighting Bias Correction)
==============================================================
论文来源: Senthilkumar et al., IJERPH 2019
公式: R_m = OBS_m / CTM_m, FC(s) = CTM(s) * IDW(R)

该方法使用IDW插值观测与模型的偏差比值
"""

import numpy as np
from scipy.spatial.distance import cdist


class IDWBias:
    """
    IDW偏差加权融合法

    核心公式: R_m = OBS_m / CTM_m
             IDW(R) = sum(w_i * R_i) / sum(w_i), w_i = 1/d_i^p
             FC(s) = CTM(s) * IDW(R)
    """
    def __init__(self, power=2.0, max_distance=100.0, min_neighbors=3):
        """
        Parameters
        ----------
        power : float
            IDW距离权重指数 (p)
        max_distance : float
            最大插值距离(km)
        min_neighbors : int
            最小近邻数
        """
        self.power = power
        self.max_distance = max_distance
        self.min_neighbors = min_neighbors

    def fit(self, X_obs, y_obs, y_model_obs):
        """
        拟合模型

        Parameters
        ----------
        X_obs : array (n, 2)
            监测站点坐标
        y_obs : array (n,)
            观测值
        y_model_obs : array (n,)
            CMAQ模型在站点的预测值
        """
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs
        self.n_obs = len(y_obs)

        # 计算偏差比值 R = OBS / CTM
        # 处理极端值
        ratio = y_obs / (y_model_obs + 1e-6)
        self.ratio = np.clip(ratio, 0.2, 5.0)  # 限制在合理范围

        # 计算站点间距离
        self.dists = cdist(X_obs, X_obs, 'euclidean')

    def predict(self, X_grid, y_model_grid):
        """
        预测

        Parameters
        ----------
        X_grid : array (n_grid, 2)
            网格点坐标
        y_model_grid : array (n_grid,)
            网格点CMAQ模型值

        Returns
        -------
        y_pred : array (n_grid,)
            融合预测值
        """
        n_grid = X_grid.shape[0]

        # 计算网格点到观测站的距离
        dists_grid_obs = cdist(X_grid, self.X_obs, 'euclidean')

        ratio_pred = np.zeros(n_grid)

        for i in range(n_grid):
            dists_i = dists_grid_obs[i]

            # 找到有效范围内的近邻
            valid_mask = dists_i <= self.max_distance
            if valid_mask.sum() < self.min_neighbors:
                # 如果近邻不足，使用全部数据中最近的几个
                idx = np.argpartition(dists_i, self.min_neighbors)[:self.min_neighbors]
            else:
                idx = np.where(valid_mask)[0]

            dists_k = dists_i[idx]
            dists_k = np.maximum(dists_k, 1e-6)

            # IDW权重
            weights = 1.0 / (dists_k ** self.power)
            weights /= weights.sum()

            ratio_pred[i] = np.sum(weights * self.ratio[idx])

        # 融合: FC = CTM * IDW(R)
        y_pred = y_model_grid * ratio_pred

        # 确保非负
        y_pred = np.maximum(y_pred, 0)

        return y_pred


class IDWBiasWeighted:
    """
    带距离加权的IDW偏差融合法

    公式: R_weighted = sum(w_i * (OBS/CTM) * (1/d_i)) / sum(w_i * (1/d_i))
    """
    def __init__(self, power=2.0, max_distance=100.0, min_neighbors=3):
        """
        Parameters
        ----------
        power : float
            IDW距离权重指数
        max_distance : float
            最大插值距离
        min_neighbors : int
            最小近邻数
        """
        self.power = power
        self.max_distance = max_distance
        self.min_neighbors = min_neighbors

    def fit(self, X_obs, y_obs, y_model_obs):
        """拟合"""
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs
        self.n_obs = len(y_obs)

        # 计算偏差比值
        ratio = y_obs / (y_model_obs + 1e-6)
        self.ratio = np.clip(ratio, 0.2, 5.0)

    def predict(self, X_grid, y_model_grid):
        """预测"""
        n_grid = X_grid.shape[0]
        dists_grid_obs = cdist(X_grid, self.X_obs, 'euclidean')

        ratio_pred = np.zeros(n_grid)

        for i in range(n_grid):
            dists_i = dists_grid_obs[i]

            # 有效范围内的近邻
            valid_mask = dists_i <= self.max_distance
            if valid_mask.sum() < self.min_neighbors:
                idx = np.argpartition(dists_i, self.min_neighbors)[:self.min_neighbors]
            else:
                idx = np.where(valid_mask)[0]

            dists_k = dists_i[idx]
            dists_k = np.maximum(dists_k, 1e-6)

            # 带距离加权的IDW
            w_i = 1.0 / (dists_k ** self.power)
            w_total = w_i * (1.0 / dists_k)  # 额外的距离加权

            weights = w_total / (w_total.sum() + 1e-6)

            ratio_pred[i] = np.sum(weights * self.ratio[idx])

        y_pred = y_model_grid * ratio_pred
        y_pred = np.maximum(y_pred, 0)

        return y_pred
