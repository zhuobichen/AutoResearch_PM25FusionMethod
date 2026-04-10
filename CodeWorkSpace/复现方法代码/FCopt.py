"""
FCopt优化融合法 (Friberg Optimized Fusion Method)
=================================================
论文来源: Friberg et al., ES&T 2016
公式: FCopt = W * FC1 + (1-W) * FC2
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


class FCoptFusion:
    """
    FCopt优化融合法（简化版）

    核心公式:
    W = R1 * (1-R2) / (R1*(1-R2) + R2*(1-R1))
    FCopt = W * FC1 + (1-W) * FC2
    """
    def __init__(self, W_min=0.0, variogram_model='exponential', k=20):
        """
        Parameters
        ----------
        W_min : float
            最小权重
        variogram_model : str
            半变异函数模型
        k : int
            近邻数量
        """
        self.W_min = W_min
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

        # 1. 拟合FC1组件
        self._fit_fc1_components()

        # 2. 计算R2 (CMAQ时间相关性)
        self._compute_R2()

    def _fit_fc1_components(self):
        """拟合FC1组件"""
        # 均值
        self.mean_obs = np.mean(self.y_obs)

        # 归一化比值
        self.ratio = self.y_obs / (self.mean_obs + 1e-6)

        # 年均值校正
        model = LinearRegression()
        X_reg = self.y_model_obs.reshape(-1, 1)
        model.fit(X_reg, self.y_obs)
        self.alpha = model.intercept_
        self.beta = model.coef_[0]

        if self.beta < 0.1:
            self.beta = 1.0

        # 全局校正比值
        self.global_ratio = np.mean(self.y_obs / (self.y_model_obs + 1e-6))

    def _compute_R2(self):
        """计算CMAQ时间相关性R2"""
        if self.n_obs > 2:
            corr, _ = pearsonr(self.y_obs, self.y_model_obs)
            self.R2 = max(0.1, min(0.9, corr))
        else:
            self.R2 = 0.5

    def predict(self, X_grid, y_model_grid):
        """
        预测
        """
        n_grid = X_grid.shape[0]

        # 1. 计算FC1
        fc1 = self._compute_fc1(X_grid, y_model_grid)

        # 2. 计算FC2
        fc2 = self._compute_fc2(X_grid, y_model_grid)

        # 3. 计算R1 (时空相关性)
        R1 = self._compute_R1(X_grid)

        # 4. 计算权重W
        W = self._compute_weight(R1)

        # 5. 融合
        y_pred = W * fc1 + (1 - W) * fc2

        # 确保非负
        y_pred = np.maximum(y_pred, 0)

        return y_pred

    def _compute_fc1(self, X_grid, y_model_grid):
        """计算FC1"""
        # IDW插值比值
        ratio_kriged = self._idw_ratio(X_grid)

        # 年均场
        annual_grid = self.alpha + self.beta * y_model_grid

        return ratio_kriged * annual_grid

    def _compute_fc2(self, X_grid, y_model_grid):
        """计算FC2"""
        return y_model_grid * self.global_ratio

    def _compute_R1(self, X_grid):
        """计算时空相关性R1"""
        n_grid = X_grid.shape[0]
        dists_grid_obs = cdist(X_grid, self.X_obs, 'euclidean')
        min_dist = np.min(dists_grid_obs, axis=1)

        # R1随距离衰减
        r = 1.0
        R1 = np.exp(-min_dist / r) * 0.5 + 0.3

        return R1

    def _compute_weight(self, R1):
        """计算融合权重W"""
        R2 = self.R2
        W_min = self.W_min

        numerator = R1 * (1 - R2)
        denominator = R1 * (1 - R2) + R2 * (1 - R1) + 1e-6

        W = numerator / denominator
        W = np.clip(W, W_min, 1 - W_min)

        return W

    def _idw_ratio(self, X_grid):
        """IDW插值比值"""
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

            weights = 1.0 / (dists_k ** 2)
            weights /= weights.sum()

            ratio_pred[i] = np.sum(weights * self.ratio[idx])

        return ratio_pred
