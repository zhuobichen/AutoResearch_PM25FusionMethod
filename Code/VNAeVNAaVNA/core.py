"""
VNA/eVNA/aVNA 核心融合逻辑
"""
import pandas as pd
import numpy as np
import multiprocessing
import sys
import os

from .nna_methods import NNA


class VNAFusionCore:
    """
    VNA/eVNA/aVNA 核心融合类
    封装了三种融合方法的计算逻辑
    """

    def __init__(self, k=30, power=-2, method='voronoi'):
        """
        初始化融合器

        参数:
            k: 近邻数量 (默认30, EPA标准)
            power: 距离权重指数 (默认-2, IDW)
            method: 邻居选择方法 (默认'voronoi', EPA标准)
        """
        self.k = k
        self.power = power
        self.method = method
        self._nn = None

    def fit(self, obs_df, monitor_col='Conc', mod_col='mod', bias_col='bias', rn_col='r_n'):
        """
        拟合 NNA 模型

        参数:
            obs_df: 监测站点数据 DataFrame
            monitor_col: 监测值列名
            mod_col: 模型值列名
            bias_col: 偏差列名
            rn_col: 比值列名
        """
        self._nn = NNA(method=self.method, k=self.k, power=self.power)
        self._nn.fit(
            obs_df[['x', 'y']] if 'x' in obs_df.columns else obs_df[['Lon', 'Lat']],
            obs_df[[monitor_col, mod_col, bias_col, rn_col]]
        )
        return self

    def predict(self, grid_coords):
        """
        预测网格点的融合值

        参数:
            grid_coords: 预测网格坐标

        返回:
            DataFrame: 含 vna, avna, evna 结果
        """
        njobs = multiprocessing.cpu_count()
        zdf = self._nn.predict(grid_coords, njobs=njobs)

        result = pd.DataFrame(
            zdf,
            columns=['vna', 'vna_mod', 'vna_bias', 'vna_rn']
        )
        return result

    @staticmethod
    def compute_avna(model_values, vna_bias):
        """
        计算 aVNA (加法修正)

        公式: O_avna = O_model + bias
        """
        return model_values + vna_bias

    @staticmethod
    def compute_evna(model_values, vna_rn):
        """
        计算 eVNA (乘法修正)

        公式: O_evna = O_model * r_n
        """
        return model_values * vna_rn
