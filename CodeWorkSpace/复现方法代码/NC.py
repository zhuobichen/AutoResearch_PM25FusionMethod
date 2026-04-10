"""
华北WRF-Chem多源融合方法(纯监测+CMAQ部分)
=========================================
基于论文《评估一种用于估算华北地区日均PM2.5浓度的数据融合方法》

方法: North China WRF-Chem Multi-source Fusion (NC)
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar


class NorthChinaFusion:
    """
    华北多源融合方法

    基于贝叶斯加权融合，将CMAQ模型与监测数据结合
    公式: P_NC = M + w * (O - M)
    其中权重w是空间的函数，通过核平滑确定
    """

    def __init__(self, kernel='gaussian', bandwidth=None, k=30, power=-2):
        """
        Parameters
        ----------
        kernel : str
            核函数类型: 'gaussian', 'epanechnikov', 'uniform'
        bandwidth : float, optional
            核带宽（度），None时自动选择
        k : int
            用于计算权重的近邻数量
        power : float
            IDW权重指数（作为备选方法）
        """
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.k = k
        self.power = power
        self.opt_bandwidth_ = None

    def fit(self, X_obs, y_obs, y_model_obs):
        """
        拟合模型

        Parameters
        ----------
        X_obs : array (n_obs, 2)
            监测站点坐标 [lon, lat]
        y_obs : array (n_obs,)
            监测站点观测值
        y_model_obs : array (n_obs,)
            监测站点对应的CMAQ模型值
        """
        self.X_obs = np.asarray(X_obs)
        self.y_obs = np.asarray(y_obs)
        self.y_model_obs = np.asarray(y_model_obs)

        # 计算残差
        self.residual = y_obs - y_model_obs

        # 自动选择最优带宽
        if self.bandwidth is None:
            self._select_bandwidth()
        else:
            self.opt_bandwidth_ = self.bandwidth

        # 估计空间范围（用于归一化距离）
        self._estimate_spatial_scale()

    def _estimate_spatial_scale(self):
        """估计空间尺度"""
        n = len(self.X_obs)
        if n < 2:
            self.spatial_scale_ = 1.0
            return

        dists = cdist(self.X_obs, self.X_obs, 'euclidean')
        idx_i, idx_j = np.triu_indices(n, k=1)
        upper_dists = dists[idx_i, idx_j]

        if len(upper_dists) > 0:
            # 使用中位数作为特征空间尺度
            self.spatial_scale_ = np.percentile(upper_dists, 50)
            if self.spatial_scale_ < 0.1:
                self.spatial_scale_ = 0.5
        else:
            self.spatial_scale_ = 0.5

    def _select_bandwidth(self):
        """使用留一法交叉验证选择最优带宽"""
        n = len(self.X_obs)
        if n < 10:
            self.opt_bandwidth_ = 0.5
            return

        # 候选带宽值
        bandwidths = np.linspace(0.1, 2.0, 10)

        # 计算每个带宽的LOO-CV误差
        cv_errors = []
        for h in bandwidths:
            errors = []
            for i in range(n):
                # 留一点
                mask = np.ones(n, dtype=bool)
                mask[i] = False

                X_train = self.X_obs[mask]
                r_train = self.residual[mask]

                # 用该带宽预测第i点
                dists = np.sqrt(np.sum((self.X_obs[i] - X_train) ** 2, axis=1))

                if self.k < len(X_train):
                    idx = np.argpartition(dists, self.k)[:self.k]
                else:
                    idx = np.arange(len(X_train))

                dists_k = dists[idx]
                r_k = r_train[idx]

                # 高斯核权重
                weights = np.exp(-0.5 * (dists_k / h) ** 2)
                weights /= (weights.sum() + 1e-10)

                # 预测残差
                r_pred = np.sum(weights * r_k)

                # 计算误差
                errors.append((self.residual[i] - r_pred) ** 2)

            cv_errors.append(np.mean(errors))

        # 选择最优带宽
        best_idx = np.argmin(cv_errors)
        self.opt_bandwidth_ = bandwidths[best_idx]

    def _kernel_weight(self, dist, h):
        """计算核权重"""
        if self.kernel == 'gaussian':
            # 高斯核
            u = dist / h
            return np.exp(-0.5 * u ** 2)
        elif self.kernel == 'epanechnikov':
            # Epanechnikov核
            u = dist / h
            return np.maximum(1 - u ** 2, 0)
        else:  # uniform
            # 均匀核
            return 1.0 if dist <= h else 0.0

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
        X_grid = np.asarray(X_grid)
        y_model_grid = np.asarray(y_model_grid)

        # 核平滑插值残差
        residual_pred = self._kernel_smoothing_residual(X_grid)

        return y_model_grid + residual_pred

    def _kernel_smoothing_residual(self, X_grid):
        """核平滑插值残差"""
        n_grid = X_grid.shape[0]
        residual_pred = np.zeros(n_grid)

        h = self.opt_bandwidth_

        for i in range(n_grid):
            x0 = X_grid[i:i+1]

            # 距离
            dists = cdist(x0, self.X_obs, 'euclidean').ravel()

            # 选择k个最近邻
            if len(dists) > self.k:
                idx = np.argpartition(dists, self.k)[:self.k]
            else:
                idx = np.arange(len(dists))

            dists_k = dists[idx]
            residual_k = self.residual[idx]

            # 核权重
            weights = self._kernel_weight(dists_k, h)

            # 归一化
            weights /= (weights.sum() + 1e-10)

            # 加权平均
            residual_pred[i] = np.sum(weights * residual_k)

        return residual_pred

    def predict_at_locations(self, X_locations, y_model):
        """
        在指定位置预测

        Parameters
        ----------
        X_locations : array (n, 2)
            位置坐标
        y_model : array (n,)
            对应的模型值

        Returns
        -------
        y_pred : array (n,)
            预测值
        """
        return self.predict(X_locations, y_model)


class NC_IDW(NorthChinaFusion):
    """
    简化版华北融合方法

    使用IDW代替核平滑，提高计算效率
    """

    def __init__(self, k=30, power=-2):
        """
        Parameters
        ----------
        k : int
            近邻数量
        power : float
            IDW权重指数
        """
        super().__init__(k=k, power=power)

    def _kernel_smoothing_residual(self, X_grid):
        """IDW插值残差"""
        n_grid = X_grid.shape[0]
        residual_pred = np.zeros(n_grid)

        for i in range(n_grid):
            x0 = X_grid[i:i+1]

            # 距离
            dists = cdist(x0, self.X_obs, 'euclidean').ravel()

            # 选择k个最近邻
            if len(dists) > self.k:
                idx = np.argpartition(dists, self.k)[:self.k]
            else:
                idx = np.arange(len(dists))

            dists_k = dists[idx]
            residual_k = self.residual[idx]

            # IDW权重
            dists_k = np.maximum(dists_k, 1e-6)
            if self.power < 0:
                weights = dists_k ** (-self.power)
            else:
                weights = 1.0 / (dists_k ** self.power)
            weights /= (weights.sum() + 1e-10)

            # 加权平均
            residual_pred[i] = np.sum(weights * residual_k)

        return residual_pred
