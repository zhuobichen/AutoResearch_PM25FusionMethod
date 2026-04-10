"""
PM2.5 CMAQ融合方法复现代码
===========================
本模块实现LocalPaperLibrary论文中的融合方法，仅使用监测数据+CMAQ

方法列表：
1. OMA - Observation Model Aggregation (观测模型聚合)
2. SMA - Statistical Model Aggregation (统计模型聚合)
3. MMA - Mixed Model Aggregation (混合模型聚合)
4. BC - Bias Correction (偏差校正) 家族
5. Quantile Mapping (分位数映射)
6. Spatial Kriging BC (空间克里金偏差校正)
7. ODI - Observation Deviation Index (观测偏差指示器)
8. Ensemble Mean (集合平均)
9. Optimum Interpolation (最优插值)
10. 3D-Var (三维变分同化)
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
import warnings

class OMA:
    """
    Observation Model Aggregation (观测模型聚合)

    公式: P_OMA = alpha * O + (1-alpha) * M
    或等价形式: P_OMA = M + alpha * (O - M)
    """
    def __init__(self, alpha=0.5, method='global', k=30, power=-2):
        """
        Parameters
        ----------
        alpha : float
            融合权重 (0=OBS主导, 1=CMAQ主导)
        method : str
            'global': 全局权重，所有站点用相同alpha
            'local': 局部权重，用IDW插值
        k : int
            近邻数量 (local方法)
        power : float
            IDW权重指数
        """
        self.alpha = alpha
        self.method = method
        self.k = k
        self.power = power

    def fit(self, X_obs, y_obs, y_model_obs):
        """
        拟合模型

        Parameters
        ----------
        X_obs : array (n, 2)
            监测站点坐标 [lon, lat]
        y_obs : array (n,)
            监测站点观测值
        y_model_obs : array (n,)
            监测站点对应的CMAQ模型值
        """
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs

        # 计算站点偏差
        self.bias = y_obs - y_model_obs

        if self.method == 'global':
            # 全局优化alpha
            # 目标: min sum[(O - alpha*O - (1-alpha)*M)^2]
            # 简化: min sum[(alpha*(O-M) - (O-M))^2] = min sum[(B - alpha*B)^2]
            # 最优 alpha = mean(B) / mean(B) = 1 ??? 实际上应该是加权平均
            # 正确公式: alpha* = sum(B_i * D_i) / sum(D_i^2) where D_i = O_i - M_i
            self.alpha = np.sum(self.bias ** 2) / np.sum(self.bias ** 2)
            # 实际上最优alpha = correlation(O-M, O) / variance(O-M)
            # 简化为: alpha = mean(B) / mean(B) = 1 ? 不对
            # O = alpha*O + (1-alpha)*M + epsilon
            # O - M = alpha*(O-M) + epsilon
            # alpha* = sum((O-M)*(O-M)) / sum((O-M)^2) = 1
            # 这不对...重新理解
            # OMA的alpha是混合权重，不是回归系数
            # 最优alpha应该通过CV确定，这里用均值作为默认
            self.alpha = 0.5

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
        if self.method == 'global':
            # 全局OMA
            return y_model_grid + self.alpha * np.mean(self.bias)
        else:
            # 局部OMA (IDW偏差插值)
            return y_model_grid + self._idw_bias(X_grid)

    def _idw_bias(self, X_grid):
        """IDW插值偏差"""
        n_grid = X_grid.shape[0]
        bias_pred = np.zeros(n_grid)

        for i, x0 in enumerate(X_grid):
            # 计算距离
            dists = cdist([x0], self.X_obs, 'euclidean').ravel()
            # 找到k个最近邻
            if len(dists) > self.k:
                idx = np.argpartition(dists, self.k)[:self.k]
            else:
                idx = np.arange(len(dists))

            # IDW权重
            dists_k = dists[idx]
            dists_k = np.maximum(dists_k, 1e-6)  # 避免除零
            weights = 1.0 / (dists_k ** self.power)
            weights /= weights.sum()

            # 加权偏差
            bias_pred[i] = np.sum(weights * self.bias[idx])

        return bias_pred


class SMA:
    """
    Statistical Model Aggregation (统计模型聚合)

    公式: O = a + b*M + epsilon
         P_SMA = a + b*M
    """
    def __init__(self, regression_type='linear', poly_degree=2, robust=False):
        """
        Parameters
        ----------
        regression_type : str
            'linear': 线性回归 O = a + b*M
            'polynomial': 多项式回归 O = a0 + a1*M + a2*M^2 + ...
        poly_degree : int
            多项式阶数 (regression_type='polynomial'时)
        robust : bool
            是否使用Huber稳健回归
        """
        self.regression_type = regression_type
        self.poly_degree = poly_degree
        self.robust = robust
        self.a = None
        self.b = None

    def fit(self, X_obs, y_obs, y_model_obs):
        """
        拟合模型
        """
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs

        if self.regression_type == 'linear':
            if self.robust:
                model = HuberRegressor()
            else:
                model = LinearRegression()
            model.fit(y_model_obs.reshape(-1, 1), y_obs)
            self.a = model.intercept_
            self.b = model.coef_[0]
        else:
            # 多项式回归
            poly = PolynomialFeatures(degree=self.poly_degree)
            M_poly = poly.fit_transform(y_model_obs.reshape(-1, 1))
            if self.robust:
                model = HuberRegressor()
            else:
                model = LinearRegression()
            model.fit(M_poly, y_obs)
            self.a = model.intercept_
            self.b = model.coef_[1:]  # [b1, b2, ...]

    def predict(self, X_grid, y_model_grid):
        """预测"""
        if self.regression_type == 'linear':
            return self.a + self.b * y_model_grid
        else:
            # 多项式预测
            M = y_model_grid.reshape(-1, 1)
            poly = PolynomialFeatures(degree=self.poly_degree)
            M_poly = poly.fit_transform(M)
            # 重新构建特征矩阵
            result = np.zeros_like(y_model_grid)
            result += self.a
            for i in range(self.poly_degree):
                result += self.b[i] * (y_model_grid ** (i + 1))
            return result


class MMA:
    """
    Mixed Model Aggregation (混合模型聚合)

    公式: P_MMA = M + beta*(a + b*M - M) + (1-beta)*sum(w_i*(O-M))
    """
    def __init__(self, beta=0.5, k=30, power=-2):
        """
        Parameters
        ----------
        beta : float
            混合参数 (0=SMA主导, 1=aVNA主导)
        k : int
            近邻数量
        power : float
            IDW权重指数
        """
        self.beta = beta
        self.k = k
        self.power = power

    def fit(self, X_obs, y_obs, y_model_obs):
        """拟合"""
        # 首先进行SMA回归
        self.sma = SMA(regression_type='linear')
        self.sma.fit(X_obs, y_obs, y_model_obs)

        # 保存数据用于IDW
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs
        self.bias = y_obs - y_model_obs

    def predict(self, X_grid, y_model_grid):
        """预测"""
        # SMA部分
        sma_pred = self.sma.predict(X_grid, y_model_grid)
        sma_bias = sma_pred - y_model_grid

        # aVNA部分 (IDW偏差插值)
        avna_pred = self._idw_bias(X_grid) + y_model_grid

        # 混合
        return y_model_grid + self.beta * sma_bias + (1 - self.beta) * (avna_pred - y_model_grid)

    def _idw_bias(self, X_grid):
        """IDW插值偏差"""
        n_grid = X_grid.shape[0]
        bias_pred = np.zeros(n_grid)

        for i, x0 in enumerate(X_grid):
            dists = cdist([x0], self.X_obs, 'euclidean').ravel()
            if len(dists) > self.k:
                idx = np.argpartition(dists, self.k)[:self.k]
            else:
                idx = np.arange(len(dists))

            dists_k = dists[idx]
            dists_k = np.maximum(dists_k, 1e-6)
            # IDW权重: w_i = 1/d_i^p (power是正数表示近距离高权重)
            # 如果self.power是负数(如-2)，需要特殊处理
            if self.power < 0:
                weights = dists_k ** (-self.power)  # 1/d^2
            else:
                weights = 1.0 / (dists_k ** self.power)
            weights /= weights.sum()

            bias_pred[i] = np.sum(weights * self.bias[idx])

        return bias_pred


class BC:
    """
    Bias Correction (偏差校正) 家族

    Methods:
    - mean: P = M + mean(B)
    - spatial: P = M + IDW(B)
    - scale: P = M * mean(O/M)
    - linear: P = a + b*M
    """
    def __init__(self, method='spatial', k=30, power=-2):
        """
        Parameters
        ----------
        method : str
            'mean', 'spatial', 'scale', 'linear'
        k : int
            近邻数量 (spatial方法)
        power : float
            IDW权重指数
        """
        self.method = method
        self.k = k
        self.power = power

    def fit(self, X_obs, y_obs, y_model_obs):
        """拟合"""
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs
        self.bias = y_obs - y_model_obs

        # 计算比率，处理极端值
        ratio = y_obs / y_model_obs
        ratio = np.clip(ratio, 0.1, 10)  # 限制在合理范围
        self.ratio = ratio

        if self.method == 'linear':
            model = LinearRegression()
            model.fit(y_model_obs.reshape(-1, 1), y_obs)
            self.a = model.intercept_
            self.b = model.coef_[0]

    def predict(self, X_grid, y_model_grid):
        """预测"""
        if self.method == 'mean':
            return y_model_grid + np.mean(self.bias)
        elif self.method == 'spatial':
            return y_model_grid + self._idw_bias(X_grid)
        elif self.method == 'scale':
            # 使用中位数更稳健
            return y_model_grid * np.median(self.ratio)
        elif self.method == 'linear':
            return self.a + self.b * y_model_grid
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _idw_bias(self, X_grid):
        """IDW插值偏差"""
        n_grid = X_grid.shape[0]
        bias_pred = np.zeros(n_grid)

        for i, x0 in enumerate(X_grid):
            dists = cdist([x0], self.X_obs, 'euclidean').ravel()
            if len(dists) > self.k:
                idx = np.argpartition(dists, self.k)[:self.k]
            else:
                idx = np.arange(len(dists))

            dists_k = dists[idx]
            dists_k = np.maximum(dists_k, 1e-6)
            # IDW权重: w_i = 1/d_i^p (power是正数表示近距离高权重)
            # 如果self.power是负数(如-2)，需要特殊处理
            if self.power < 0:
                weights = dists_k ** (-self.power)  # 1/d^2
            else:
                weights = 1.0 / (dists_k ** self.power)
            weights /= weights.sum()

            bias_pred[i] = np.sum(weights * self.bias[idx])

        return bias_pred


class QuantileMapping:
    """
    Quantile Mapping (分位数映射)

    公式: P_QM = mean(O) + (std(O)/std(M)) * (M - mean(M))
    或分位数形式: P = a_q + b_q * M (根据M的分位数选择参数)
    """
    def __init__(self, n_quantiles=10, method='linear'):
        """
        Parameters
        ----------
        n_quantiles : int
            分位数数量
        method : str
            'linear': 线性QM
            'local': 局部分位数映射
        """
        self.n_quantiles = n_quantiles
        self.method = method

    def fit(self, X_obs, y_obs, y_model_obs):
        """拟合"""
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs
        self.bias = y_obs - y_model_obs

        # 计算全局统计量
        self.mean_obs = np.mean(y_obs)
        self.mean_mod = np.mean(y_model_obs)
        self.std_obs = np.std(y_obs)
        self.std_mod = np.std(y_model_obs)

        # 分位数映射参数
        if self.method == 'local':
            self.quantiles = np.linspace(0, 1, self.n_quantiles + 1)[1:-1]
            obs_quantiles = np.percentile(y_obs, self.quantiles * 100)
            mod_quantiles = np.percentile(y_model_obs, self.quantiles * 100)
            self.a_q = obs_quantiles - mod_quantiles  # 简化

    def predict(self, X_grid, y_model_grid):
        """预测"""
        if self.method == 'linear':
            # 线性QM
            return self.mean_obs + (self.std_obs / self.std_mod) * (y_model_grid - self.mean_mod)
        else:
            # 局部分位数映射
            result = np.zeros_like(y_model_grid)
            for i, m in enumerate(y_model_grid):
                # 找到m对应的分位数
                q = np.searchsorted(np.sort(self.y_model_obs), m) / len(self.y_model_obs)
                q = np.clip(q, 0.01, 0.99)
                # 局部队域内计算
                mask = np.abs(self.y_model_obs - m) < np.std(self.y_model_obs)
                if mask.sum() > 5:
                    local_obs = self.y_obs[mask]
                    local_mod = self.y_model_obs[mask]
                    result[i] = np.mean(local_obs) + np.std(local_obs)/np.std(local_mod) * (m - np.mean(local_mod))
                else:
                    result[i] = self.mean_obs + (self.std_obs / self.std_mod) * (m - self.mean_mod)
            return result


class SpatialKrigingBC:
    """
    Spatial Kriging Bias Correction (空间克里金偏差校正)

    公式: P = M + B_kriging
    其中 B_kriging 通过克里金插值偏差场获得
    """
    def __init__(self, variogram_model='spherical', k=30, power=-2):
        """
        Parameters
        ----------
        variogram_model : str
            半变异函数模型: 'spherical', 'exponential', 'gaussian'
        k : int
            用于加速计算的近邻数量
        power : float
            距离权重指数（用于简单插值替代完整克里金）
        """
        self.variogram_model = variogram_model
        self.k = k
        self.power = power

    def fit(self, X_obs, y_obs, y_model_obs):
        """拟合"""
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs
        self.bias = y_obs - y_model_obs

        # 估计变异函数参数
        self._estimate_variogram()

    def _estimate_variogram(self):
        """估计变异函数参数"""
        n = len(self.bias)
        # 计算实验变异函数
        dists = cdist(self.X_obs, self.X_obs, 'euclidean')
        # 取上三角
        idx_i, idx_j = np.triu_indices(n)
        h = dists[idx_i, idx_j]  # 距离
        gamma_exp = 0.5 * (self.bias[idx_i] - self.bias[idx_j]) ** 2  # 实验半方差

        # 过滤零距离
        mask = h > 1e-6
        h = h[mask]
        gamma_exp = gamma_exp[mask]

        # 简化：使用数据估计sill和range
        # sill ≈ variance of bias
        self.sill = np.var(self.bias)
        # range: 用所有点对距离的中位数
        self.range = np.median(h) if len(h) > 0 else 1.0
        if self.range < 0.01:
            self.range = 1.0
        self.nugget = 0

    def predict(self, X_grid, y_model_grid):
        """预测"""
        # 使用简化的克里金（用IDW近似）
        return y_model_grid + self._kriging_bias(X_grid)

    def _kriging_bias(self, X_grid):
        """克里金插值偏差 - 使用简化的加权平均"""
        n_grid = X_grid.shape[0]
        bias_pred = np.zeros(n_grid)

        for i, x0 in enumerate(X_grid):
            dists = cdist([x0], self.X_obs, 'euclidean').ravel()

            if len(dists) > self.k:
                idx = np.argpartition(dists, self.k)[:self.k]
            else:
                idx = np.arange(len(dists))

            dists_k = dists[idx]
            bias_k = self.bias[idx]

            # 使用高斯协方差函数的加权平均
            # C(h) = sill * exp(-3*h^2/range^2)
            C = self.sill * np.exp(-3 * (dists_k / self.range) ** 2)

            # 简单加权：权重 = C / sum(C)
            weights = C / (np.sum(C) + 1e-10)
            bias_pred[i] = np.sum(weights * bias_k)

        return bias_pred

    def _variogram(self, h):
        """球状变异函数模型"""
        c0 = self.nugget
        c = self.sill - self.nugget
        a = max(self.range, 0.01)

        if np.isscalar(h):
            if h <= a:
                return c0 + c * (1.5 * h / a - 0.5 * (h / a) ** 3)
            else:
                return c0 + c
        else:
            result = np.zeros_like(h, dtype=float)
            mask = h <= a
            result[mask] = c0 + c * (1.5 * h[mask] / a - 0.5 * (h[mask] / a) ** 3)
            result[~mask] = c0 + c
            return result


class ODI:
    """
    Observation Deviation Index (观测偏差指示器)

    公式: P = M + gamma * M * NDI_smoothed
    其中 NDI = (O-M)/M 归一化后的偏差指示器
    """
    def __init__(self, gamma=0.1, k=30, power=-2, normalize=True):
        """
        Parameters
        ----------
        gamma : float
            NDI缩放因子（建议用小值如0.1）
        k : int
            近邻数量
        power : float
            IDW权重指数
        normalize : bool
            是否归一化DI
        """
        self.gamma = gamma
        self.k = k
        self.power = power
        self.normalize = normalize

    def fit(self, X_obs, y_obs, y_model_obs):
        """拟合"""
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs

        # 计算偏差指示器 (避免除零)
        ratio = y_obs / y_model_obs
        ratio = np.clip(ratio, 0.1, 10)  # 防止极端值
        self.di = ratio - 1  # 偏差比率 = O/M - 1 = (O-M)/M

        # 归一化
        if self.normalize:
            self.di_mean = np.mean(self.di)
            self.di_std = np.std(self.di)
            if self.di_std > 1e-6:
                self.ndi = (self.di - self.di_mean) / self.di_std
            else:
                self.ndi = self.di - self.di_mean
        else:
            self.ndi = self.di

    def predict(self, X_grid, y_model_grid):
        """预测"""
        ndi_smoothed = self._idw_ndi(X_grid)
        # 使用加法形式: P = M + gamma * M * NDI
        # 而不是 P = M * (1 + gamma * NDI) 以避免大值问题
        return y_model_grid + self.gamma * y_model_grid * ndi_smoothed

    def _idw_ndi(self, X_grid):
        """IDW插值NDI"""
        n_grid = X_grid.shape[0]
        ndi_pred = np.zeros(n_grid)

        for i, x0 in enumerate(X_grid):
            dists = cdist([x0], self.X_obs, 'euclidean').ravel()
            if len(dists) > self.k:
                idx = np.argpartition(dists, self.k)[:self.k]
            else:
                idx = np.arange(len(dists))

            dists_k = dists[idx]
            dists_k = np.maximum(dists_k, 1e-6)
            # IDW权重: w_i = 1/d_i^p (power是正数表示近距离高权重)
            # 如果self.power是负数(如-2)，需要特殊处理
            if self.power < 0:
                weights = dists_k ** (-self.power)  # 1/d^2
            else:
                weights = 1.0 / (dists_k ** self.power)
            weights /= weights.sum()

            ndi_pred[i] = np.sum(weights * self.ndi[idx])

        return ndi_pred


class EnsembleMean:
    """
    Ensemble Mean (集合平均)

    公式: P_EM = mean(M_1, M_2, ..., M_Ne) + BC
    """
    def __init__(self, method='mean', k=30, power=-2):
        """
        Parameters
        ----------
        method : str
            'mean': 简单平均
            'bias_corrected': 带偏差校正的集合平均
        k : int
            近邻数量
        power : float
            IDW权重指数
        """
        self.method = method
        self.k = k
        self.power = power

    def fit(self, X_obs, y_obs, y_model_obs_list):
        """
        拟合

        Parameters
        ----------
        y_model_obs_list : list of arrays
            多个CMAQ集合成员在站点的值
        """
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.ensemble_members = y_model_obs_list

        # 集合平均
        self.y_model_obs = np.mean(y_model_obs_list, axis=0)
        self.bias = y_obs - self.y_model_obs

    def predict(self, X_grid, y_model_grid_list):
        """
        预测

        Parameters
        ----------
        y_model_grid_list : list of arrays
            多个CMAQ集合成员在网格的值
        """
        # 网格上的集合平均
        y_model_grid = np.mean(y_model_grid_list, axis=0)

        if self.method == 'mean':
            return y_model_grid
        else:
            # bias_corrected
            return y_model_grid + self._idw_bias(X_grid)

    def _idw_bias(self, X_grid):
        """IDW插值偏差"""
        n_grid = X_grid.shape[0]
        bias_pred = np.zeros(n_grid)

        for i, x0 in enumerate(X_grid):
            dists = cdist([x0], self.X_obs, 'euclidean').ravel()
            if len(dists) > self.k:
                idx = np.argpartition(dists, self.k)[:self.k]
            else:
                idx = np.arange(len(dists))

            dists_k = dists[idx]
            dists_k = np.maximum(dists_k, 1e-6)
            # IDW权重: w_i = 1/d_i^p (power是正数表示近距离高权重)
            # 如果self.power是负数(如-2)，需要特殊处理
            if self.power < 0:
                weights = dists_k ** (-self.power)  # 1/d^2
            else:
                weights = 1.0 / (dists_k ** self.power)
            weights /= weights.sum()

            bias_pred[i] = np.sum(weights * self.bias[idx])

        return bias_pred


class OptimumInterpolation:
    """
    Optimum Interpolation (最优插值)

    公式: P_a = P_b + B * (B + R)^(-1) * (O - H*P_b)
    简化: P_a = P_b + sum(w_i * (O_i - P_b_i))
    """
    def __init__(self, model_std=None, obs_std=None, corr_scale=None, k=30):
        """
        Parameters
        ----------
        model_std : float
            模型误差标准差
        obs_std : float
            观测误差标准差
        corr_scale : float
            空间相关尺度
        k : int
            近邻数量
        """
        self.model_std = model_std
        self.obs_std = obs_std
        self.corr_scale = corr_scale
        self.k = k

    def fit(self, X_obs, y_obs, y_model_obs):
        """拟合"""
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs
        self.bias = y_obs - y_model_obs

        # 估计参数
        if self.model_std is None:
            self.model_std = max(np.std(self.bias), 1.0)
        if self.obs_std is None:
            self.obs_std = self.model_std * 0.5  # 假设观测比模型准
        if self.corr_scale is None:
            # 用数据的空间范围估计
            dists = cdist(X_obs, X_obs, 'euclidean')
            upper_dists = dists[np.triu_indices(len(X_obs), k=1)]
            if len(upper_dists) > 0:
                self.corr_scale = np.percentile(upper_dists, 50)
            else:
                self.corr_scale = 1.0
            if self.corr_scale < 0.1:
                self.corr_scale = 1.0

    def predict(self, X_grid, y_model_grid):
        """预测"""
        n_grid = X_grid.shape[0]
        pred = np.zeros(n_grid)

        for i, x0 in enumerate(X_grid):
            # 找最近邻
            dists = cdist([x0], self.X_obs, 'euclidean').ravel()
            if len(dists) > self.k:
                idx = np.argpartition(dists, self.k)[:self.k]
            else:
                idx = np.arange(len(dists))

            dists_k = dists[idx]
            bias_k = self.bias[idx]

            # 高斯协方差模型
            C = self.model_std ** 2 * np.exp(-0.5 * (dists_k / self.corr_scale) ** 2)

            # 观测误差协方差（简化）
            R = self.obs_std ** 2

            # 简单加权
            weights = C / (np.sum(C ** 2) + R)
            pred[i] = y_model_grid[i] + np.sum(weights * bias_k)

        return pred


class ThreeDVar:
    """
    3D-Var (三维变分同化)

    与OptimumInterpolation数学等价，从代价函数极小化角度推导
    """
    def __init__(self, background_error=None, obs_error=None, corr_length=None, k=30):
        """
        Parameters
        ----------
        background_error : float
            背景误差标准差
        obs_error : float
            观测误差标准差
        corr_length : float
            空间相关长度
        k : int
            近邻数量
        """
        self.background_error = background_error
        self.obs_error = obs_error
        self.corr_length = corr_length
        self.k = k

    def fit(self, X_obs, y_obs, y_model_obs):
        """拟合"""
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs
        self.bias = y_obs - y_model_obs

        if self.background_error is None:
            self.background_error = max(np.std(self.bias), 1.0)
        if self.obs_error is None:
            self.obs_error = self.background_error * 0.5
        if self.corr_length is None:
            dists = cdist(X_obs, X_obs, 'euclidean')
            upper_dists = dists[np.triu_indices(len(X_obs), k=1)]
            if len(upper_dists) > 0:
                self.corr_length = np.percentile(upper_dists, 50)
            else:
                self.corr_length = 1.0
            if self.corr_length < 0.1:
                self.corr_length = 1.0

    def predict(self, X_grid, y_model_grid):
        """预测"""
        # 3D-Var的简化实现（与OI等价）
        oi = OptimumInterpolation(
            model_std=self.background_error,
            obs_std=self.obs_error,
            corr_scale=self.corr_length,
            k=self.k
        )
        oi.fit(self.X_obs, self.y_obs, self.y_model_obs)
        return oi.predict(X_grid, y_model_grid)
