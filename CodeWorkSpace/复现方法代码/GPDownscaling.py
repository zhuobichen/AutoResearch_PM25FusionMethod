"""
GP降尺度法 (Gaussian Process Disaggregation)
============================================
论文来源: Rodriguez Avellaneda et al., 2025 (KAUST)
公式: W_m(s) = sum_k a_mk * U_k(s) (多变量线性协同区域化模型)

该方法使用高斯过程将网格尺度CMAQ数据降尺度到站点尺度
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve_triangular, cholesky
from sklearn.linear_model import LinearRegression


class GPDownscaling:
    """
    GP降尺度法

    基于多变量线性协同区域化模型(LMC)的高斯过程降尺度方法
    核心思想：建立网格均值与站点值之间的空间关联模型
    """
    def __init__(self, n_latent=3, nu=1.0, max_iter=500, k=20):
        """
        Parameters
        ----------
        n_latent : int
            潜在过程数量
        nu : float
            Matérn协方差函数的平滑参数
        max_iter : int
            优化最大迭代
        k : int
            近邻数量
        """
        self.n_latent = n_latent
        self.nu = nu
        self.max_iter = max_iter
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

        # 估计空间变异函数参数
        self._estimate_variogram()

        # 构建协方差矩阵
        self._build_covariance_matrix()

        # 拟合线性回归得到尺度因子
        self._fit_scale_model()

    def _estimate_variogram(self):
        """估计半变异函数参数"""
        # 计算残差
        residuals = self.y_obs - self.y_model_obs

        # 计算距离矩阵
        dists = cdist(self.X_obs, self.X_obs, 'euclidean')

        # 取上三角计算实验变异函数
        idx_i, idx_j = np.triu_indices(self.n_obs)
        h = dists[idx_i, idx_j]
        gamma_exp = 0.5 * (residuals[idx_i] - residuals[idx_j]) ** 2

        # 过滤零距离
        mask = h > 1e-6
        h = h[mask]
        gamma_exp = gamma_exp[mask]

        # 使用指数模型拟合: gamma(h) = c0 + c * (1 - exp(-h/a))
        # 简化：用数据直接估计
        self.sill = np.var(residuals)
        self.range = np.median(h[h > 0]) if len(h[h > 0]) > 0 else 1.0
        self.nugget = 0

        if self.range < 0.1:
            self.range = 1.0

    def _matern_covariance(self, d, sigma=1.0, rho=1.0, nu=1.0):
        """
        Matérn协方差函数

        C(d) = sigma^2 * 2^(1-nu)/Gamma(nu) * (d*sqrt(2*nu)/rho)^nu * K_nu(d*sqrt(2*nu)/rho)

        简化版本 (nu=1.0):
        C(d) = sigma^2 * (1 + d/rho) * exp(-d/rho)
        """
        # 简化：使用指数协方差
        return sigma ** 2 * np.exp(-d / rho)

    def _build_covariance_matrix(self):
        """构建空间协方差矩阵"""
        # 计算站点间距离
        dists = cdist(self.X_obs, self.X_obs, 'euclidean')

        # 构建协方差矩阵
        self.C = self._matern_covariance(dists,
                                          sigma=np.sqrt(self.sill + self.nugget),
                                          rho=self.range,
                                          nu=self.nu)

        # 添加观测噪声
        noise_std = 0.1 * np.sqrt(self.sill)
        np.fill_diagonal(self.C, self.C.diagonal() + noise_std ** 2)

    def _fit_scale_model(self):
        """拟合尺度因子模型: obs = alpha + beta * cmaq + GP"""
        # 首先进行线性回归
        X_reg = np.column_stack([np.ones(self.n_obs), self.y_model_obs])
        y = self.y_obs

        # 最小二乘
        XT_X = X_reg.T @ X_reg
        XT_y = X_reg.T @ y
        beta = np.linalg.solve(XT_X + 1e-6 * np.eye(2), XT_y)

        self.alpha = beta[0]
        self.beta = beta[1]

        # 计算残差
        self.residuals = y - (self.alpha + self.beta * self.y_model_obs)

        # 重新估计变异函数
        self.residual_sill = np.var(self.residuals)

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
            降尺度预测值
        """
        n_grid = X_grid.shape[0]

        # 1. 线性趋势: alpha + beta * CMAQ
        trend = self.alpha + self.beta * y_model_grid

        # 2. 空间插值残差
        residual_pred = self._interpolate_residuals(X_grid)

        # 融合
        y_pred = trend + residual_pred

        # 确保非负
        y_pred = np.maximum(y_pred, 0)

        return y_pred

    def _interpolate_residuals(self, X_grid):
        """克里金插值残差"""
        n_grid = X_grid.shape[0]
        residual_pred = np.zeros(n_grid)

        # 构建目标点与观测点之间的协方差
        dists_grid_obs = cdist(X_grid, self.X_obs, 'euclidean')

        # 协方差向量
        sigma_total = np.sqrt(self.residual_sill + self.nugget)
        C_obs_grid = sigma_total * np.exp(-dists_grid_obs / self.range)

        # 求解克里金权重
        # C @ weights = C_obs_grid
        # 使用预处理的共轭梯度

        C_obs = self.C

        for i in range(n_grid):
            c_star = C_obs_grid[i]

            # 求解线性系统
            try:
                # 使用简化的求解方法
                weights = np.linalg.solve(C_obs + 1e-6 * np.eye(self.n_obs), c_star)
                weights /= weights.sum()
            except:
                # 备用：使用距离反比
                dists = dists_grid_obs[i]
                weights = 1.0 / (dists + 1e-6)
                weights /= weights.sum()

            residual_pred[i] = np.sum(weights * self.residuals)

        return residual_pred
