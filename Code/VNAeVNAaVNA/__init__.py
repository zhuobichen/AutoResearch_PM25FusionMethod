"""
VNA/eVNA/aVNA 数据融合封装模块

提供简洁的输入入口，支持 OfficialInput 格式数据

使用示例:
    from vna_fusion import VNAFusion

    fusion = VNAFusion(
        model_file='path/to/model.nc',
        monitor_file='path/to/monitor.csv',
        region_file='path/to/region.csv'
    )

    # 处理指定某一天
    result = fusion.process_day('2020-01-01')

    # 处理日期范围
    results = fusion.process_range('2020-01-01', '2020-12-31')
"""
import pandas as pd
import numpy as np
import sys
import os
from tqdm.auto import tqdm

# 导入内部模块
from .input_handler import OfficialInput
from .core import VNAFusionCore


class VNAFusion:
    """
    VNA/eVNA/aVNA 数据融合主类

    输入数据:
        - 监测数据 (CSV): Site, Date, Conc, Lat, Lon
        - 模型数据 (NetCDF): 365天数据
            - IOAPI 格式: Time x Layer x ROW x COL
            - CF 格式: time x y x x (带 lat, lon 坐标)
        - 区域掩码 (CSV): ROW, COL, Is (可选)

    输出:
        - VNA: Voronoi邻居IDW插值
        - aVNA: 加法修正 (O_model - bias)
        - eVNA: 乘法修正 (O_model * r_n)
    """

    def __init__(self, model_file, monitor_file, region_file=None,
                 k=30, power=-2, method='voronoi',
                 monitor_pollutant='Conc', model_pollutant=None):
        """
        初始化 VNAFusion

        参数:
            model_file: NetCDF 模型文件路径
            monitor_file: CSV 监测数据文件路径
            region_file: CSV 区域掩码文件路径 (可选)
            k: 近邻数量 (默认30, EPA标准)
            power: 距离权重指数 (默认-2, IDW)
            method: 邻居选择方法 (默认'voronoi', EPA标准)
            monitor_pollutant: 监测数据浓度列名 (默认'Conc')
            model_pollutant: 模型变量名 (默认自动检测)
        """
        self.model_file = model_file
        self.monitor_file = monitor_file
        self.region_file = region_file
        self.monitor_pollutant = monitor_pollutant

        # 加载数据
        self._input = OfficialInput(
            model_file=model_file,
            monitor_file=monitor_file,
            region_file=region_file
        )
        self._input.load_monitor()
        self._input.load_model(model_pollutant=model_pollutant)
        self._input.load_region()

        # 获取模型格式和变量名
        self._model_format = self._input._model_format
        self.model_pollutant = self._input._model_pollutant

        # 融合器
        self._core = VNAFusionCore(k=k, power=power, method=method)

    @property
    def dates(self):
        """获取所有可用日期"""
        return self._input.dates

    @property
    def model_format(self):
        """获取模型数据格式 ('ioapi' 或 'cf')"""
        return self._model_format

    def process_day(self, date):
        """
        处理指定某一天

        参数:
            date: str, 'YYYY-MM-DD' 格式

        返回:
            DataFrame: 含 ROW, COL, vna, avna, evna, model, Timestamp
        """
        # 获取当日数据
        daily_data = self._input.get_daily_data(date)
        obs_df = daily_data['obs']
        model_ds = daily_data['daily_model']

        # 获取模型值并计算融合结果
        model_values = model_ds[self.model_pollutant].values
        model_flat = model_values.flatten()

        if self._model_format == 'ioapi':
            # IOAPI 格式处理
            self._core.fit(
                obs_df,
                monitor_col=self.monitor_pollutant,
                mod_col='mod',
                bias_col='bias',
                rn_col='r_n'
            )
            grid_coords = self._input.get_grid_coords()
            result = self._core.predict(grid_coords)

            # 添加网格坐标
            result['ROW'] = grid_coords[:, 1]
            result['COL'] = grid_coords[:, 0]

        else:
            # CF 格式处理 - 使用经纬度进行插值
            # 获取模型网格的经纬度坐标
            lat_grid = self._input._lat  # (y, x)
            lon_grid = self._input._lon

            # 构建预测网格坐标 (使用经纬度)
            ny, nx = lat_grid.shape
            predict_coords = np.column_stack([
                lon_grid.flatten(),  # 经度
                lat_grid.flatten()   # 纬度
            ])

            # 构建监测站点坐标
            obs_coords = np.column_stack([obs_df['Lon'].values, obs_df['Lat'].values])

            # 创建观测数据 DataFrame 用于拟合
            obs_for_fit = obs_df.copy()
            obs_for_fit['x'] = obs_df['Lon'].values
            obs_for_fit['y'] = obs_df['Lat'].values

            self._core.fit(
                obs_for_fit,
                monitor_col=self.monitor_pollutant,
                mod_col='mod',
                bias_col='bias',
                rn_col='r_n'
            )

            # 预测
            result = self._core.predict(predict_coords)

            # 添加网格坐标 (ROW, COL)
            result['ROW'] = np.repeat(np.arange(ny), nx)
            result['COL'] = np.tile(np.arange(nx), ny)

        # 计算 aVNA 和 eVNA
        result['avna'] = VNAFusionCore.compute_avna(model_flat, result['vna_bias'].values)
        result['evna'] = VNAFusionCore.compute_evna(model_flat, result['vna_rn'].values)
        result['model'] = model_flat
        result['Timestamp'] = date

        # 整理列顺序
        result = result[['ROW', 'COL', 'vna', 'avna', 'evna', 'model', 'Timestamp']]

        return result

    def process_range(self, start_date, end_date):
        """
        处理日期范围

        参数:
            start_date: str, 开始日期 'YYYY-MM-DD'
            end_date: str, 结束日期 'YYYY-MM-DD'

        返回:
            DataFrame: 所有日期的融合结果
        """
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        all_results = []

        for date in tqdm(all_dates, desc="数据融合"):
            date_str = date.strftime('%Y-%m-%d')
            try:
                result = self.process_day(date_str)
                all_results.append(result)
            except Exception as e:
                print(f"处理 {date_str} 时出错: {e}")
                continue

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()

    def save(self, result_df, output_file):
        """
        保存结果到 CSV

        参数:
            result_df: 融合结果 DataFrame
            output_file: 输出文件路径
        """
        result_df.to_csv(output_file, index=False)
        print(f"结果已保存到: {output_file}")
