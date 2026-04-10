"""
GenFriberg广义融合法 (Generalized Friberg Data Fusion)
=======================================================
论文来源: Li et al., Environmental Modelling and Software 2025
公式: FC1 = krig(OBS/CMAQ) * FC, FC2 = CTM_adj * beta_season
       FC_final = W * FC1 + (1-W) * FC2
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


class GenFribergFusion:
    """
    GenFriberg广义融合法（简化版）

    简化策略：使用简单的IDW比值插值代替复杂克里金
    """
    def __init__(self, regression_mode='auto', variogram_model='exponential',
                 n_folds=10, k=20):
        """
        Parameters
        ----------
        regression_mode : str
            'linear', 'exponential', 或 'auto'
        variogram_model : str
            半变异函数模型
        n_folds : int
            交叉验证折数
        k : int
            近邻数量
        """
        self.regression_mode = regression_mode
        self.variogram_model = variogram_model
        self.n_folds = n_folds
        self.k = k

    def fit(self, X_obs, y_obs, y_model_obs):
        """
        拟合模型
        """
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs
        self.n_obs = len(y_obs)

        # 步骤1: 年均值校正回归
        self._fit_annual_regression()

        # 步骤2: 计算CMAQ-观测相关性R2
        self._compute_R2()

    def _fit_annual_regression(self):
        """年均值校正回归: OBS = alpha + beta * CMAQ"""
        X_reg = self.y_model_obs.reshape(-1, 1)
        y = self.y_obs

        model = LinearRegression()
        model.fit(X_reg, y)
        self.alpha = model.intercept_
        self.beta = model.coef_[0]

        # 确保beta为正
        if self.beta < 0.1:
            self.beta = 1.0

        # 计算年均场
        self.fc_annual = self.alpha + self.beta * self.y_model_obs

        # 计算归一化比值: OBS / CMAQ
        self.ratio = self.y_obs / (self.y_model_obs + 1e-6)
        self.ratio = np.clip(self.ratio, 0.3, 3.0)

        # 计算mean ratio用于校正
        self.mean_ratio = np.mean(self.ratio)

    def _compute_R2(self):
        """计算CMAQ-观测相关性R2"""
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

        # FC1: IDW插值比值 * 年均校正CMAQ
        fc1 = self._compute_fc1(X_grid, y_model_grid)

        # FC2: 年均校正 * CMAQ/CMAQ_annual
        fc2 = self._compute_fc2(X_grid, y_model_grid)

        # R1: 时空相关性 (简化)
        R1 = self._compute_R1(X_grid)

        # 计算权重
        W = self._compute_weight(R1)

        # 最终融合
        y_pred = W * fc1 + (1 - W) * fc2

        # 确保非负
        y_pred = np.maximum(y_pred, 0)

        return y_pred

    def _compute_fc1(self, X_grid, y_model_grid):
        """FC1融合"""
        # IDW插值归一化比值
        ratio_kriged = self._idw_ratio(X_grid)

        # 年均场校正
        annual_grid = self.alpha + self.beta * y_model_grid

        return ratio_kriged * annual_grid

    def _compute_fc2(self, X_grid, y_model_grid):
        """FC2融合: CTM_adj * beta_season"""
        # 调整CMAQ: CTM_adj = CMAQ * (FC_annual / CMAQ_annual)
        # 简化: CTM_adj = CMAQ * mean(ratio)

        return y_model_grid * self.mean_ratio

    def _compute_R1(self, X_grid):
        """计算时空相关性R1"""
        n_grid = X_grid.shape[0]
        dists_grid_obs = cdist(X_grid, self.X_obs, 'euclidean')

        # 到最近观测的距离
        min_dist = np.min(dists_grid_obs, axis=1)

        # R1随距离衰减
        r = 1.0  # 距离尺度
        R1 = np.exp(-min_dist / r) * 0.5 + 0.3  # 简化

        return R1

    def _compute_weight(self, R1):
        """计算融合权重W"""
        R2 = self.R2

        # W = R1 * (1-R2) / (R1*(1-R2) + R2*(1-R1))
        numerator = R1 * (1 - R2)
        denominator = R1 * (1 - R2) + R2 * (1 - R1) + 1e-6

        W = numerator / denominator
        W = np.clip(W, 0.1, 0.9)

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

            # IDW权重
            weights = 1.0 / (dists_k ** 2)
            weights /= weights.sum()

            ratio_pred[i] = np.sum(weights * self.ratio[idx])

        return ratio_pred
