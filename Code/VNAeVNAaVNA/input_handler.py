"""
OfficialInput 格式数据解析模块
支持 365 天监测数据和模型数据的加载
"""
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path


class OfficialInput:
    """
    OfficialInput 格式数据加载器

    监测数据格式 (CSV):
        Site, Date, Conc, Lat, Lon
        1003A, 2020-01-01, 34.625, 39.9289, 116.4174

    模型数据格式 (NetCDF):
        - IOAPI 格式: Time, Layer, ROW, COL
        - CF 格式: time, y, x (带 lat, lon 坐标)
    """

    def __init__(self, model_file, monitor_file, region_file=None):
        """
        初始化 OfficialInput 数据加载器

        参数:
            model_file: NetCDF 模型文件路径
            monitor_file: CSV 监测数据文件路径
            region_file: CSV 区域掩码文件路径 (可选)
        """
        self.model_file = Path(model_file)
        self.monitor_file = Path(monitor_file)
        self.region_file = Path(region_file) if region_file else None

        self._model_ds = None
        self._monitor_df = None
        self._region_df = None
        self._proj = None
        self._model_format = None
        self._lat = None
        self._lon = None

    def load_all(self):
        """加载所有数据"""
        self.load_monitor()
        self.load_model()
        self.load_region()
        return self

    def load_model(self, model_pollutant=None):
        """
        加载 NetCDF 模型数据 (支持 IOAPI 和 CF 格式)

        参数:
            model_pollutant: 模型变量名，默认自动选择第一个变量
        """
        import pyrsig
        import xarray as xr

        # 尝试 IOAPI 格式
        try:
            self._model_ds = pyrsig.open_ioapi(str(self.model_file))
            self._proj = pyproj.Proj(self._model_ds.crs_proj4)
            self._model_format = 'ioapi'
        except KeyError:
            # 回退到 CF 格式
            self._model_ds = xr.open_dataset(str(self.model_file))
            self._model_format = 'cf'
            # 从数据中提取经纬度
            if 'lat' in self._model_ds and 'lon' in self._model_ds:
                self._lat = self._model_ds['lat'].values
                self._lon = self._model_ds['lon'].values

        # 自动选择模型变量
        if model_pollutant is None:
            data_vars = list(self._model_ds.data_vars)
            if data_vars:
                # 优先选择包含 'PM25' 或 'O3' 的变量
                for var in data_vars:
                    if 'PM25' in var.upper() or 'O3' in var.upper():
                        model_pollutant = var
                        break
                else:
                    # 没有找到则使用第一个
                    model_pollutant = data_vars[0]
            else:
                raise ValueError("模型数据集中没有找到任何变量")

        self._model_pollutant = model_pollutant
        return self

    def load_monitor(self, monitor_pollutant='Conc'):
        """
        加载 CSV 监测数据

        参数:
            monitor_pollutant: 监测数据浓度列名，默认 'Conc'
        """
        df = pd.read_csv(self.monitor_file)

        # 过滤掉经纬度为 NaN 的站点
        df = df.dropna(subset=['Lat', 'Lon'])

        # 按站点和日期聚合
        self._monitor_df = (
            df.groupby(["Site", "Date"])
            .aggregate({
                monitor_pollutant: "mean",
                "Lat": "mean",
                "Lon": "mean"
            })
            .sort_values(by="Date")
            .reset_index()
        )
        self._monitor_pollutant = monitor_pollutant
        return self

    def load_region(self):
        """加载区域掩码数据"""
        if self.region_file and self.region_file.exists():
            self._region_df = pd.read_csv(self.region_file)
            # 筛选 Is == 1 的区域
            self._region_df = self._region_df[self._region_df['Is'] == 1]
            self._region_df[['COL', 'ROW']] = self._region_df[['COL', 'ROW']] - 0.5
        return self

    @property
    def dates(self):
        """获取所有可用日期"""
        if self._monitor_df is None:
            self.load_monitor()
        return self._monitor_df['Date'].unique()

    @property
    def model_coords(self):
        """获取模型网格坐标"""
        if self._model_ds is None:
            self.load_model()
        return self._model_ds

    def get_daily_data(self, date):
        """
        获取指定日期的监测和模型数据

        参数:
            date: str, 'YYYY-MM-DD' 格式

        返回:
            dict: 包含 'obs' (监测) 和 'model' (模型) 的字典
        """
        from .esil.date_helper import get_day_of_year

        # 获取当日监测数据
        df_obs = self._monitor_df[self._monitor_df["Date"] == date].copy()

        if self._model_format == 'ioapi':
            # IOAPI 格式处理
            df_obs["x"], df_obs["y"] = self._proj(df_obs["Lon"], df_obs["Lat"])

            # 获取当日模型数据
            if isinstance(self._model_ds['TSTEP'].values[0], np.int64):
                time_index = get_day_of_year(date) - 1
                ds_daily = self._model_ds.sel(TSTEP=time_index)
            else:
                ds_daily = self._model_ds.sel(TSTEP=date)

            # 提取模型值到监测站点位置
            df_obs["mod"] = ds_daily[self._model_pollutant][0].sel(
                ROW=df_obs["y"].to_xarray(),
                COL=df_obs["x"].to_xarray(),
                method="nearest"
            )

            model_result = ds_daily
        else:
            # CF 格式处理 - 直接使用经纬度最近邻
            from scipy.spatial import cKDTree

            # 获取当日模型数据 (time 索引)
            if 'time' in self._model_ds.dims:
                # 尝试将日期字符串转换为时间索引
                date_obj = pd.to_datetime(date)
                time_values = self._model_ds['time'].values

                # 如果 time 是天数索引 (0-364)
                if np.issubdtype(time_values.dtype, np.integer):
                    time_index = (date_obj.dayofyear - 1) % 365
                else:
                    time_index = date_obj
                ds_daily = self._model_ds.isel(time=time_index)
            else:
                ds_daily = self._model_ds

            # 构建监测站点的经纬度树
            obs_coords = np.column_stack([df_obs['Lon'].values, df_obs['Lat'].values])

            # 构建模型网格的经纬度树
            model_lon_flat = self._lon.flatten()
            model_lat_flat = self._lat.flatten()
            model_grid_coords = np.column_stack([model_lon_flat, model_lat_flat])
            tree = cKDTree(model_grid_coords)

            # 找到每个监测站点最近的模型网格点
            distances, indices = tree.query(obs_coords)
            model_values = ds_daily[self._model_pollutant].values
            model_values_flat = model_values.flatten()
            df_obs["mod"] = model_values_flat[indices]

            # CF 格式不使用 x, y，使用索引代替
            df_obs["grid_idx"] = indices

            model_result = self._model_ds

        # 计算 bias 和 ratio
        df_obs["bias"] = df_obs["mod"] - df_obs[self._monitor_pollutant]
        df_obs["r_n"] = df_obs[self._monitor_pollutant] / df_obs["mod"]

        return {
            "obs": df_obs,
            "model": model_result,
            "daily_model": ds_daily,
            "model_pollutant": self._model_pollutant,
            "format": self._model_format
        }

    def get_grid_coords(self):
        """
        获取预测网格坐标

        返回:
            ndarray: 格式取决于模型格式
                - IOAPI: [COL, ROW]
                - CF: [grid_idx]
        """
        if self._region_df is not None:
            return self._region_df[['COL', 'ROW']].values

        if self._model_format == 'ioapi':
            # 返回全部 IOAPI 网格
            df = self._model_ds[["ROW", "COL"]].to_dataframe().reset_index()
            df[['COL', 'ROW']] = df[['COL', 'ROW']] - 0.5
            return df[['COL', 'ROW']].values
        else:
            # CF 格式: 返回扁平化的网格索引
            n_points = self._lat.size
            return np.arange(n_points)

    def get_model_shape(self):
        """获取模型数据的空间形状"""
        if self._model_format == 'ioapi':
            return self._model_ds[self._model_pollutant][0].shape
        else:
            return self._model_ds[self._model_pollutant].shape[1:]  # (y, x)
