"""
FC1克里金插值融合法 (Friberg FC1 Method)
========================================
论文来源: Friberg et al., ES&T 2016
公式: FC1(s,t) = krig(OBS(t)/OBS) * FC_annual(s)
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression


class FC1Kriging:
    """
    FC1克里金插值融合法（简化版）

    核心公式:
    FC1(s,t) = krig(R) * FC_annual(s)
    其中 R = OBS(t) / mean(OBS) 是归一化比值
    """
    def __init__(self, variogram_model='exponential', k=20):
        """
        Parameters
        ----------
        variogram_model : str
            半变异函数模型
        k : int
            近邻数量
        """
        self.variogram_model = variogram_model
        self.k = k

    def fit(self, X_obs, y_obs, y_model_obs):
        """
        拟合模型
        """
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs
        self.n_obs = len(y_obs)

        # 计算均值
        self.mean_obs = np.mean(y_obs)

        # 计算归一化比值 R = OBS / mean(OBS)
        self.ratio = y_obs / (self.mean_obs + 1e-6)

        # 拟合CMAQ年均场校正参数
        self._fit_annual_correction()

    def _fit_annual_correction(self):
        """CMAQ年均场校正: OBS_mean = alpha + beta * CMAQ_mean"""
        X_reg = self.y_model_obs.reshape(-1, 1)
        y = self.y_obs

        model = LinearRegression()
        model.fit(X_reg, y)
        self.alpha = model.intercept_
        self.beta = model.coef_[0]

        # 确保beta为正
        if self.beta < 0.1:
            self.beta = 1.0

    def predict(self, X_grid, y_model_grid):
        """
        预测
        """
        n_grid = X_grid.shape[0]

        # 1. 克里金插值归一化比值 (简化使用IDW)
        ratio_kriged = self._idw_ratio(X_grid)

        # 2. 校正后的年均场
        annual_grid = self.alpha + self.beta * y_model_grid

        # 3. FC1 = krig(R) * annual_CMAQ
        y_pred = ratio_kriged * annual_grid

        # 确保非负
        y_pred = np.maximum(y_pred, 0)

        return y_pred

    def _idw_ratio(self, X_grid):
        """IDW插值归一化比值"""
        n_grid = X_grid.shape[0]
        ratio_pred = np.zeros(n_grid)

        dists_grid_obs = cdist(X_grid, self.X_obs, 'euclidean')

        for i in range(n_grid):
            dists_i = dists_grid_obs[i]

            if self.n_obs > self.k:
                idx = np.argpartition(dists_i, self.k)[:self.k]
            else:
                idx = np.arange(self.n_obs)

            dists_k = dists_i[idx]
            dists_k = np.maximum(dists_k, 1e-6)

            # IDW权重
            weights = 1.0 / (dists_k ** 2)
            weights /= weights.sum()

            ratio_pred[i] = np.sum(weights * self.ratio[idx])

        return ratio_pred
