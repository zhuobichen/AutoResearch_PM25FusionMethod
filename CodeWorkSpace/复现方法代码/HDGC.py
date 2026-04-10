"""
HDGC监测偏差检测法 (Hidden Dynamic Geostatistical Calibration)
==============================================================
论文来源: Wang et al., 2019 (arXiv:1901.03939)
公式: Z(s,t) = mu(s) + U(s,t), Y_i(t) = gamma_i * Z(s_i,t) + epsilon_i(t)

该方法检测PM2.5监测站的系统性偏差
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve_triangular


class HDGC:
    """
    HDGC监测偏差检测法

    使用隐藏动态地统计模型检测监测站偏差
    核心思想：每个站点有一个校准参数gamma_i
    """
    def __init__(self, max_iter=200, tol=1e-4, rho_s=None, rho_t=None, k=10):
        """
        Parameters
        ----------
        max_iter : int
            EM算法最大迭代
        tol : float
            收敛容忍度
        rho_s : float
            空间相关尺度
        rho_t : float
            时间相关尺度
        k : int
            空间近邻数
        """
        self.max_iter = max_iter
        self.tol = tol
        self.rho_s = rho_s
        self.rho_t = rho_t
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

        # 估计空间尺度
        if self.rho_s is None:
            dists = cdist(X_obs, X_obs, 'euclidean')
            upper_dists = dists[np.triu_indices(self.n_obs, k=1)]
            self.rho_s = np.percentile(upper_dists, 50) if len(upper_dists) > 0 else 1.0

        # 初始化校准参数
        self.gamma = np.ones(self.n_obs)

        # 估计隐藏场
        self._em_algorithm()

        # 计算偏差检测统计量
        self._compute_bias_statistics()

    def _em_algorithm(self):
        """EM算法估计"""
        n = self.n_obs

        # 计算站点间距离
        dists = cdist(self.X_obs, self.X_obs, 'euclidean')

        # 时空协方差结构（简化：只用空间）
        # Cov(U_i, U_j) = sigma^2 * exp(-d_ij / rho_s)
        sigma2 = np.var(self.y_obs - self.y_model_obs)

        # 构建协方差矩阵
        C = sigma2 * np.exp(-dists / self.rho_s)
        np.fill_diagonal(C, C.diagonal() + 1e-6)  # 确保正定

        # EM迭代
        for iteration in range(self.max_iter):
            # E步：估计隐藏场（真值）
            # Z = gamma * CMAQ + C @ (C + R)^(-1) * (obs - gamma * CMAQ)
            # 简化：使用线性混合效应模型

            # 残差
            residuals = self.y_obs - self.gamma * self.y_model_obs

            # 简单更新：gamma_new = obs / CMAQ 的加权平均
            ratio = self.y_obs / (self.y_model_obs + 1e-6)
            ratio = np.clip(ratio, 0.5, 2.0)  # 限制在合理范围

            # 使用空间平滑更新gamma
            gamma_new = np.ones(n)
            for i in range(n):
                # 找到近邻
                dist_row = dists[i]
                idx = np.argpartition(dist_row, min(self.k, n-1))[:min(self.k, n-1)]

                # 高斯权重
                weights = np.exp(-0.5 * (dist_row[idx] / self.rho_s) ** 2)
                weights /= weights.sum()

                gamma_new[i] = np.sum(weights * ratio[idx])

            # 检查收敛
            diff = np.max(np.abs(gamma_new - self.gamma))
            self.gamma = gamma_new

            if diff < self.tol:
                break

    def _compute_bias_statistics(self):
        """计算偏差检测统计量"""
        # 每个站点的gamma标准误（简化估计）
        n = self.n_obs

        # 残差
        residuals = self.y_obs - self.gamma * self.y_model_obs

        # 标准误估计
        self.gamma_se = np.std(residuals) / (np.abs(self.y_model_obs) + 1e-6)

        # 偏差标记
        self.bias_flag = np.abs(self.gamma - 1) > 2 * self.gamma_se

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
            校正后的预测值
        """
        n_grid = X_grid.shape[0]

        # 对网格点插值gamma
        gamma_pred = self._interpolate_gamma(X_grid)

        # 校正CMAQ
        y_pred = gamma_pred * y_model_grid

        # 确保非负
        y_pred = np.maximum(y_pred, 0)

        return y_pred

    def _interpolate_gamma(self, X_grid):
        """克里金插值gamma"""
        n_grid = X_grid.shape[0]
        gamma_pred = np.zeros(n_grid)

        dists = cdist(X_grid, self.X_obs, 'euclidean')

        for i in range(n_grid):
            dist_row = dists[i]
            idx = np.argpartition(dist_row, min(self.k, self.n_obs-1))[:min(self.k, self.n_obs-1)]

            dists_k = dist_row[idx]
            dists_k = np.maximum(dists_k, 1e-6)

            # 高斯权重
            weights = np.exp(-0.5 * (dists_k / self.rho_s) ** 2)
            weights /= weights.sum()

            gamma_pred[i] = np.sum(weights * self.gamma[idx])

        return gamma_pred
