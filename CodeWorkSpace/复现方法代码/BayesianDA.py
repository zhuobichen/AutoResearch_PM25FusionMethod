"""
贝叶斯数据同化法 (Bayesian Data Assimilation)
============================================
论文来源: Chianese et al., Ecological Modelling 2018
公式: J(b) = (F(x-b) - y)^T P^{-1}(F(x-b) - y) + (b - b_0)^T Q^{-1}(b - b_0)

该方法使用正则化最小二乘估计CTM偏差场
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression


class BayesianDA:
    """
    贝叶斯数据同化法（简化版）

    核心思想：偏差场由平滑约束的正则化最小二乘估计
    """
    def __init__(self, omega=1.0, epsilon=1e-2, delta=1e-2, max_iter=100, tol=1e-6):
        """
        Parameters
        ----------
        omega : float
            观测拟合权重
        epsilon : float
            平滑正则化参数
        delta : float
            小偏差权重
        """
        self.omega = omega
        self.epsilon = epsilon
        self.delta = delta

    def fit(self, X_obs, y_obs, y_model_obs):
        """
        拟合模型
        """
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs
        self.n_obs = len(y_obs)

        # 计算初始偏差 (CMAQ - OBS)
        self.bias = y_model_obs - y_obs

        # 构建空间平滑矩阵
        self._build_smoothing_matrix()

        # 求解正则化最小二乘
        self._solve_regularized_ls()

    def _build_smoothing_matrix(self):
        """构建平滑约束矩阵"""
        n = self.n_obs
        dists = cdist(self.X_obs, self.X_obs, 'euclidean')

        # 使用高斯核构建平滑矩阵
        sigma = np.median(dists[dists > 0]) * 0.5
        self.S = np.exp(-0.5 * (dists / sigma) ** 2)
        np.fill_diagonal(self.S, 1)

    def _solve_regularized_ls(self):
        """求解正则化最小二乘"""
        n = self.n_obs

        # 构建系统矩阵
        # J(b) = omega * ||b - bias||^2 + epsilon * b^T S^T S b + delta * ||b||^2
        # 导数为零: 2*omega*(b - bias) + 2*epsilon*S^T*S*b + 2*delta*b = 0
        # (omega*I + epsilon*S^T*S + delta*I) * b = omega * bias

        STS = self.S.T @ self.S
        A = self.omega * np.eye(n) + self.epsilon * STS + self.delta * np.eye(n)
        b_rhs = self.omega * self.bias

        try:
            self.bias_est = np.linalg.solve(A + 1e-6 * np.eye(n), b_rhs)
        except:
            self.bias_est = self.bias

    def predict(self, X_grid, y_model_grid):
        """
        预测
        """
        n_grid = X_grid.shape[0]

        # 插值偏差到网格
        bias_pred = self._interpolate_bias(X_grid)

        # 融合: y = CMAQ - bias
        y_pred = y_model_grid - bias_pred

        # 确保非负
        y_pred = np.maximum(y_pred, 0)

        return y_pred

    def _interpolate_bias(self, X_grid):
        """IDW插值偏差"""
        n_grid = X_grid.shape[0]
        bias_pred = np.zeros(n_grid)

        dists_grid_obs = cdist(X_grid, self.X_obs, 'euclidean')

        for i in range(n_grid):
            dists_i = dists_grid_obs[i]
            k = min(15, self.n_obs)
            idx = np.argpartition(dists_i, k)[:k]

            dists_k = dists_i[idx]
            dists_k = np.maximum(dists_k, 1e-6)

            # 高斯权重
            sigma = np.median(dists_k) * 0.5
            if sigma < 1e-6:
                sigma = 1.0
            weights = np.exp(-0.5 * (dists_k / sigma) ** 2)
            weights /= weights.sum()

            bias_pred[i] = np.sum(weights * self.bias_est[idx])

        return bias_pred
