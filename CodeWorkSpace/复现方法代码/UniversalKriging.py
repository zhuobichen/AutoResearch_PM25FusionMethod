"""
通用克里金PM25映射法 (Universal Kriging)
=========================================
论文来源: Berrocal et al., 2019 (arXiv:1904.08931)
公式: Y(s) = X(s)*beta + U(s), Y_hat(s_0) = X(s_0)*beta_hat + sum_i lambda_i*(Y(s_i) - X(s_i)*beta_hat)

该方法使用CMAQ作为协变量的克里金插值
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve_triangular, cholesky
from sklearn.linear_model import LinearRegression


class UniversalKriging:
    """
    通用克里金PM25映射法

    使用CMAQ作为趋势项的克里金插值
    核心思想: Y(s) = X(s)*beta + U(s), 其中U(s)是空间随机场
    """
    def __init__(self, variogram_model='exponential', drift='linear', k=20):
        """
        Parameters
        ----------
        variogram_model : str
            半变异函数模型: 'exponential', 'spherical', 'gaussian'
        drift : str
            趋势函数类型: 'linear', 'quadratic'
        k : int
            近邻数量
        """
        self.variogram_model = variogram_model
        self.drift = drift
        self.k = k

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

        # 1. 趋势建模: 使用CMAQ作为协变量拟合回归
        self._fit_trend()

        # 2. 计算残差
        self.residuals = self.y_obs - self.trend_pred

        # 3. 拟合半变异函数
        self._fit_variogram()

    def _fit_trend(self):
        """拟合趋势模型"""
        # X(s) = [1, CMAQ(s)]
        self.X_trend = np.column_stack([np.ones(self.n_obs), self.y_model_obs])

        # 最小二乘拟合
        XT_X = self.X_trend.T @ self.X_trend
        XT_y = self.X_trend.T @ self.y_obs
        self.beta = np.linalg.solve(XT_X + 1e-6 * np.eye(2), XT_y)

        self.trend_pred = self.X_trend @ self.beta

    def _fit_variogram(self):
        """拟合半变异函数"""
        n = self.n_obs

        # 计算距离矩阵
        dists = cdist(self.X_obs, self.X_obs, 'euclidean')

        # 取上三角计算实验变异函数
        idx_i, idx_j = np.triu_indices(n)
        h = dists[idx_i, idx_j]
        gamma_exp = 0.5 * (self.residuals[idx_i] - self.residuals[idx_j]) ** 2

        # 过滤零距离
        mask = h > 1e-6
        h = h[mask]
        gamma_exp = gamma_exp[mask]

        # 估计参数
        self.nugget = 0
        self.sill = np.var(self.residuals)

        # 使用指数模型拟合range
        # gamma(h) = c0 + c * (1 - exp(-h/a))
        if len(h) > 0:
            # 简化的range估计
            self.range_param = np.median(h)
            if self.range_param < 0.1:
                self.range_param = 1.0
        else:
            self.range_param = 1.0

    def _variogram_model(self, h):
        """半变异函数模型"""
        c0 = self.nugget
        c = self.sill - self.nugget
        a = max(self.range_param, 0.01)

        if self.variogram_model == 'exponential':
            # gamma(h) = c0 + c * (1 - exp(-h/a))
            return c0 + c * (1 - np.exp(-h / a))
        elif self.variogram_model == 'spherical':
            # gamma(h) = c0 + c * (1.5*h/a - 0.5*(h/a)^3) for h <= a
            result = np.zeros_like(h, dtype=float)
            mask = h <= a
            result[mask] = c0 + c * (1.5 * h[mask] / a - 0.5 * (h[mask] / a) ** 3)
            result[~mask] = c0 + c
            return result
        elif self.variogram_model == 'gaussian':
            # gamma(h) = c0 + c * (1 - exp(-h^2/a^2))
            return c0 + c * (1 - np.exp(-(h / a) ** 2))
        else:
            return c0 + c * (1 - np.exp(-h / a))

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

        # 1. 计算趋势项
        X_trend_grid = np.column_stack([np.ones(n_grid), y_model_grid])
        trend_grid = X_trend_grid @ self.beta

        # 2. 克里金插值残差
        residual_kriging = self._kriging_residuals(X_grid)

        # 3. 合并
        y_pred = trend_grid + residual_kriging

        # 确保非负
        y_pred = np.maximum(y_pred, 0)

        return y_pred

    def _kriging_residuals(self, X_grid):
        """克里金插值残差"""
        n_grid = X_grid.shape[0]
        residual_pred = np.zeros(n_grid)

        # 计算观测点之间的距离
        dists_obs = cdist(self.X_obs, self.X_obs, 'euclidean')

        # 半变异函数矩阵 Gamma
        Gamma = self._variogram_model(dists_obs)

        # 添加约束矩阵
        # 通用克里金需要满足无偏约束
        # 解方程: [Gamma, X; X^T, 0] [lambda; mu] = [gamma_0; x_0]

        for i in range(n_grid):
            # 计算到所有观测点的距离
            dists_to_obs = cdist([X_grid[i]], self.X_obs, 'euclidean').ravel()

            # gamma_0: 待预测点到所有观测点的半方差
            gamma_0 = self._variogram_model(dists_to_obs)

            # x_0: 趋势协变量
            x_0 = np.array([1, self.y_model_obs[i]])  # 注意：使用站点i的CMAQ值

            # 构建系统矩阵 - 通用克里金方程
            # [Gamma   X] [lambda]   [gamma_0]
            # [X^T     0] [mu    ] = [x_0    ]
            p = 2  # 趋势参数数量
            A = np.zeros((self.n_obs + p, self.n_obs + p))
            A[:self.n_obs, :self.n_obs] = Gamma
            A[:self.n_obs, self.n_obs:] = self.X_trend  # (n_obs, p)
            A[self.n_obs:, :self.n_obs] = self.X_trend.T  # (p, n_obs)

            # 右侧向量
            b = np.zeros(self.n_obs + p)
            b[:self.n_obs] = gamma_0
            b[self.n_obs:] = x_0  # (p,)

            # 求解
            try:
                weights = np.linalg.solve(A + 1e-6 * np.eye(self.n_obs + p), b)
                lambda_i = weights[:self.n_obs]

            except:
                # 备用：使用简单IDW
                idx = np.argpartition(dists_to_obs, min(self.k, self.n_obs-1))[:min(self.k, self.n_obs-1)]
                lambda_i = np.zeros(self.n_obs)
                lambda_i[idx] = 1.0 / (dists_to_obs[idx] + 1e-6)
                lambda_i /= lambda_i.sum()

            residual_pred[i] = np.sum(lambda_i * self.residuals)

        return residual_pred
