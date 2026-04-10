"""
FC2尺度CMAQ融合法 (Friberg FC2 Scaled CMAQ Method)
==================================================
论文来源: Friberg et al., ES&T 2016
公式: FC2(s,t) = CMAQ(s,t) * beta * FC_annual(s) / CMAQ_annual(s) * beta_season(t)
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression


class FC2ScaleCMAQ:
    """
    FC2尺度CMAQ融合法（简化版）

    核心公式:
    FC2(s,t) = CMAQ(s,t) * correction(s)

    特点:
    - 不依赖空间稀疏的观测网络
    - 使用站点校正因子进行空间插值
    """
    def __init__(self, seasonal_correction=True, k=20):
        """
        Parameters
        ----------
        seasonal_correction : bool
            是否应用季节校正
        k : int
            近邻数量
        """
        self.seasonal_correction = seasonal_correction
        self.k = k

    def fit(self, X_obs, y_obs, y_model_obs):
        """
        拟合模型
        """
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs
        self.n_obs = len(y_obs)

        # 步骤1: 年度回归
        self._fit_annual_regression()

        # 步骤2: 季节校正因子（简化）
        if self.seasonal_correction:
            self._fit_seasonal_correction()
        else:
            self.seasonal_factor = 1.0

    def _fit_annual_regression(self):
        """年均值场回归: OBS = alpha + beta * CMAQ"""
        X_reg = self.y_model_obs.reshape(-1, 1)
        y = self.y_obs

        model = LinearRegression()
        model.fit(X_reg, y)
        self.alpha = model.intercept_
        self.beta = model.coef_[0]

        # 确保beta为正
        if self.beta < 0.1:
            self.beta = 1.0

        # 校正后的年均场
        self.fc_annual = self.alpha + self.beta * self.y_model_obs

        # 计算CMAQ年均值
        self.cmaq_annual = self.y_model_obs

        # 比值 ratio = FC_annual / CMAQ_annual
        self.ratio = self.fc_annual / (self.cmaq_annual + 1e-6)
        self.ratio = np.clip(self.ratio, 0.5, 2.0)

    def _fit_seasonal_correction(self):
        """季节校正因子（简化版本）"""
        # 使用常数因子
        global_ratio = np.mean(self.y_obs / (self.y_model_obs + 1e-6))
        self.seasonal_factor = np.clip(global_ratio, 0.8, 1.2)

    def predict(self, X_grid, y_model_grid):
        """
        预测
        """
        n_grid = X_grid.shape[0]

        # 1. 插值校正因子
        correction = self._interpolate_ratio(X_grid) * self.seasonal_factor

        # 2. FC2 = CMAQ * correction
        y_pred = y_model_grid * correction

        # 确保非负
        y_pred = np.maximum(y_pred, 0)

        return y_pred

    def _interpolate_ratio(self, X_grid):
        """克里金插值年场比值"""
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

            # 高斯权重
            sigma = np.median(dists_k) * 0.5
            if sigma < 1e-6:
                sigma = 1.0
            weights = np.exp(-0.5 * (dists_k / sigma) ** 2)
            weights /= weights.sum()

            ratio_pred[i] = np.sum(weights * self.ratio[idx])

        return ratio_pred
