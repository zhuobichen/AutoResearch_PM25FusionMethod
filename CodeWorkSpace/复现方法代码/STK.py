"""
北京STK时空克里金方法
=====================
基于论文《基于时空克里金模型估算北京地区日均PM2.5暴露量》

方法: Spatiotemporal Kriging Bias Correction
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar


class SpatioTemporalKriging:
    """
    时空克里金偏差校正方法

    公式: P_STK = M + R_kriging
    其中 R_kriging 通过时空克里金插值残差场获得
    """

    def __init__(self, spatial_model='spherical', temporal_model='exponential',
                 k=30, spatial_range=None, temporal_range=None,
                 nugget=0, verbose=False):
        """
        Parameters
        ----------
        spatial_model : str
            空间变异函数模型: 'spherical', 'exponential', 'gaussian'
        temporal_model : str
            时间变异函数模型: 'exponential', 'linear'
        k : int
            用于加速计算的近邻数量
        spatial_range : float, optional
            空间相关距离（度），None时自动估计
        temporal_range : float, optional
            时间相关距离（天），None时自动估计
        nugget : float
            块金效应值
        verbose : bool
            是否输出详细信息
        """
        self.spatial_model = spatial_model
        self.temporal_model = temporal_model
        self.k = k
        self.spatial_range = spatial_range
        self.temporal_range = temporal_range
        self.nugget = nugget
        self.verbose = verbose

    def fit(self, X_obs, y_obs, y_model_obs, time_obs=None):
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
        time_obs : array (n_obs,), optional
            监测站点对应的时间（天数），用于时空克里金
        """
        self.X_obs = np.asarray(X_obs)
        self.y_obs = np.asarray(y_obs)
        self.y_model_obs = np.asarray(y_model_obs)
        self.time_obs = np.asarray(time_obs) if time_obs is not None else None

        # 计算残差
        self.residual = y_obs - y_model_obs

        # 估计时空变异函数参数
        self._estimate_variogram_parameters()

        if self.verbose:
            print(f"Spatial range: {self.spatial_range:.4f} deg")
            print(f"Temporal range: {self.temporal_range:.4f} days")
            print(f"Spatial sill: {self.spatial_sill:.4f}")
            print(f"Temporal sill: {self.temporal_sill:.4f}")

    def _estimate_variogram_parameters(self):
        """估计时空变异函数参数"""
        n = len(self.residual)

        # ===== 空间变异函数参数 =====
        if self.spatial_range is None:
            # 计算站点间距离
            spatial_dists = cdist(self.X_obs, self.X_obs, 'euclidean')
            # 取上三角距离（避免重复）
            idx_i, idx_j = np.triu_indices(n, k=1)
            h_spatial = spatial_dists[idx_i, idx_j]

            # 过滤极端值
            h_spatial = h_spatial[h_spatial > 0.01]
            if len(h_spatial) > 0:
                # 使用半方差估计range
                self.spatial_range = np.percentile(h_spatial, 50)
                if self.spatial_range < 0.1:
                    self.spatial_range = 0.5  # 默认值
            else:
                self.spatial_range = 0.5

        # 空间sill = 残差方差
        self.spatial_sill = np.var(self.residual)
        if self.spatial_sill < 1e-6:
            self.spatial_sill = 1.0

        # ===== 时间变异函数参数 =====
        if self.time_obs is not None and self.temporal_range is None:
            # 计算时间距离
            time_diffs = []
            for i in range(n):
                for j in range(i+1, n):
                    time_diffs.append(abs(self.time_obs[i] - self.time_obs[j]))

            if len(time_diffs) > 0:
                time_diffs = np.array(time_diffs)
                self.temporal_range = np.percentile(time_diffs, 50)
                if self.temporal_range < 1:
                    self.temporal_range = 7.0  # 默认值（周周期）
            else:
                self.temporal_range = 7.0
        elif self.temporal_range is None:
            self.temporal_range = 7.0

        self.temporal_sill = self.spatial_sill  # 简化：时空sill相同

    def _spatial_variogram(self, h):
        """空间变异函数"""
        c0 = self.nugget
        c = self.spatial_sill - c0
        a = max(self.spatial_range, 0.01)

        if np.isscalar(h):
            if self.spatial_model == 'spherical':
                if h <= a:
                    return c0 + c * (1.5 * h / a - 0.5 * (h / a) ** 3)
                else:
                    return c0 + c
            elif self.spatial_model == 'exponential':
                return c0 + c * (1 - np.exp(-3 * h / a))
            elif self.spatial_model == 'gaussian':
                return c0 + c * (1 - np.exp(-3 * (h / a) ** 2))
            else:
                return c0 + c * (h > 0)
        else:
            result = np.zeros_like(h, dtype=float)
            mask = h > 0
            if self.spatial_model == 'spherical':
                m = (h <= a) & mask
                result[m] = c0 + c * (1.5 * h[m] / a - 0.5 * (h[m] / a) ** 3)
                result[~m] = c0 + c
            elif self.spatial_model == 'exponential':
                result[mask] = c0 + c * (1 - np.exp(-3 * h[mask] / a))
            elif self.spatial_model == 'gaussian':
                result[mask] = c0 + c * (1 - np.exp(-3 * (h[mask] / a) ** 2))
            return result

    def _temporal_variogram(self, tau):
        """时间变异函数"""
        c0 = 0
        c = self.temporal_sill - c0
        a = max(self.temporal_range, 0.1)

        if np.isscalar(tau):
            if self.temporal_model == 'exponential':
                return c0 + c * (1 - np.exp(-3 * tau / a))
            else:  # linear
                return c0 + c * min(tau / a, 1)
        else:
            result = np.zeros_like(tau, dtype=float)
            mask = tau > 0
            if self.temporal_model == 'exponential':
                result[mask] = c0 + c * (1 - np.exp(-3 * tau[mask] / a))
            else:
                result[mask] = c0 + c * np.minimum(tau[mask] / a, 1)
            return result

    def _joint_variogram(self, h, tau):
        """联合时空变异函数（分离型）"""
        return self._spatial_variogram(h) + self._temporal_variogram(tau)

    def _covariance(self, h, tau):
        """协方差函数 C(h,tau) = sill - gamma(h,tau)"""
        total_sill = self.spatial_sill + self.temporal_sill - np.var(self.residual)
        if total_sill <= 0:
            total_sill = self.spatial_sill
        return total_sill - self._joint_variogram(h, tau)

    def predict(self, X_grid, y_model_grid, time_grid=None):
        """
        预测

        Parameters
        ----------
        X_grid : array (n_grid, 2)
            网格点坐标
        y_model_grid : array (n_grid,)
            网格点CMAQ模型值
        time_grid : array (n_grid,), optional
            网格点对应的时间

        Returns
        -------
        y_pred : array (n_grid,)
            融合预测值
        """
        X_grid = np.asarray(X_grid)
        y_model_grid = np.asarray(y_model_grid)

        # 使用时空克里金插值残差
        residual_pred = self._kriging_residual(X_grid, time_grid)

        return y_model_grid + residual_pred

    def _kriging_residual(self, X_grid, time_grid=None):
        """克里金插值残差"""
        n_grid = X_grid.shape[0]
        residual_pred = np.zeros(n_grid)

        for i in range(n_grid):
            x0 = X_grid[i:i+1]

            # 计算到所有监测站点的距离
            dists = cdist(x0, self.X_obs, 'euclidean').ravel()

            # 时间距离（如果有）
            if time_grid is not None and self.time_obs is not None:
                time_dists = np.abs(time_grid[i] - self.time_obs)
            else:
                time_dists = np.zeros(len(self.X_obs))

            # 选择k个最近邻
            if len(dists) > self.k:
                idx = np.argpartition(dists, self.k)[:self.k]
            else:
                idx = np.arange(len(dists))

            dists_k = dists[idx]
            time_k = time_dists[idx]
            residual_k = self.residual[idx]

            # 计算协方差
            C = self._covariance(dists_k, time_k)

            # 简单克里金权重
            weights = C / (np.sum(C) + 1e-10)

            # 加权平均
            residual_pred[i] = np.sum(weights * residual_k)

        return residual_pred

    def predict_at_locations(self, X_locations, y_model, time=None):
        """
        在指定位置预测

        Parameters
        ----------
        X_locations : array (n, 2)
            位置坐标
        y_model : array (n,)
            对应的模型值
        time : array (n,), optional
            对应的时间

        Returns
        -------
        y_pred : array (n,)
            预测值
        """
        return self.predict(X_locations, y_model, time)


class SimpleSTK(SpatioTemporalKriging):
    """
    简化版时空克里金

    使用高斯核加权代替完整克里金系统求解，提高计算效率
    """

    def __init__(self, k=30, spatial_scale=0.5, temporal_scale=7.0, power=-2):
        """
        Parameters
        ----------
        k : int
            近邻数量
        spatial_scale : float
            空间相关尺度（度）
        temporal_scale : float
            时间相关尺度（天）
        power : float
            距离权重指数
        """
        super().__init__(k=k, spatial_range=spatial_scale,
                        temporal_range=temporal_scale)
        self.power = power

    def _kriging_residual(self, X_grid, time_grid=None):
        """使用高斯核加权插值残差"""
        n_grid = X_grid.shape[0]
        residual_pred = np.zeros(n_grid)

        # 计算空间协方差矩阵
        for i in range(n_grid):
            x0 = X_grid[i:i+1]

            # 空间距离
            dists = cdist(x0, self.X_obs, 'euclidean').ravel()

            # 时间距离
            if time_grid is not None and self.time_obs is not None:
                time_dists = np.abs(time_grid[i] - self.time_obs)
            else:
                time_dists = np.zeros(len(self.X_obs))

            # 组合距离
            combined_dists = np.sqrt(
                (dists / max(self.spatial_range, 0.01)) ** 2 +
                (time_dists / max(self.temporal_range, 0.1)) ** 2
            )

            # 选择k个最近邻
            if len(dists) > self.k:
                idx = np.argpartition(combined_dists, self.k)[:self.k]
            else:
                idx = np.arange(len(dists))

            # 高斯核权重
            weights = np.exp(-3 * combined_dists[idx] ** 2)
            weights /= (weights.sum() + 1e-10)

            # 加权平均残差
            residual_pred[i] = np.sum(weights * self.residual[idx])

        return residual_pred
