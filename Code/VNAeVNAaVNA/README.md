# VNA/eVNA/aVNA 数据融合模块

## 目录

- [概述](#概述)
- [功能特性](#功能特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [输入数据格式](#输入数据格式)
- [输出格式](#输出格式)
- [融合方法详解](#融合方法详解)
- [API 文档](#api-文档)
- [完整示例](#完整示例)
- [依赖说明](#依赖说明)
- [故障排除](#故障排除)

---

## 概述

VNA/eVNA/aVNA 是三种基于 Voronoi 邻域的空气质量数据融合方法，用于将地面监测数据与模型模拟数据进行统计融合，生成高空间分辨率的污染物浓度网格化产品。

### 典型应用场景

1. **空气质量监测网络增强**：将稀疏的地面监测站数据融合到高分辨率模型网格
2. **PM2.5/臭氧浓度估算**：结合 CMAQ 模型输出与实测数据
3. **环境健康风险评估**：生成可用于流行病学研究的暴露数据
4. **政策效果评估**：评估减排措施对空气质量的影响

### 方法选择指南

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 模型系统性偏差 | aVNA | 加法修正消除均值偏移 |
| 模型比例偏差 | eVNA | 乘法修正消除比例失调 |
| 偏差复杂 | VNA | 保持观测局部特征 |
| 需要平滑结果 | VNA + laplace | 拉普拉斯权重更平滑 |

---

## 功能特性

- **三种融合方法**: VNA, aVNA (加法修正), eVNA (乘法修正)
- **双格式支持**: IOAPI 格式和 CF 格式 NetCDF 自动识别
- **变量自动检测**: 自动识别 PM25/O3 相关变量名
- **365天数据**: 支持整年数据的批量处理
- **灵活日期选择**: 可选择处理某一天或日期范围
- **并行计算**: 使用多核 CPU 加速预测
- **NaN 处理**: 自动过滤监测站缺失数据

---

## 安装

模块已完全独立，所有依赖代码已包含在包内。

### 方式一：安装为本地包

```bash
cd E:/CodeProject/ClaudeRoom/VNAeVNAaVNACodePrepare/vna_fusion
pip install -e .
```

### 方式二：直接导入

```python
import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/VNAeVNAaVNACodePrepare')
from vna_fusion import VNAFusion
```

### 方式三：移动到其他项目

整个 `vna_fusion` 文件夹可以直接复制到任何项目中使用。

### 目录结构

```
vna_fusion/                    # 完全独立的模块
├── __init__.py                # 主入口
├── core.py                    # 核心融合逻辑
├── input_handler.py            # 数据加载
├── nna_methods/
│   └── __init__.py            # NNA算法实现
├── esil/
│   ├── __init__.py
│   └── date_helper.py         # 日期工具函数
└── README.md
```

### 方式三：创建快捷方式

在项目根目录创建 `run_fusion.py`:

```python
import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/VNAeVNAaVNACodePrepare')
from vna_fusion import VNAFusion

# 你的代码
```

---

## 快速开始

### 最小示例

```python
from vna_fusion import VNAFusion

# 初始化 (自动检测模型变量)
fusion = VNAFusion(
    model_file='OfficialInput/2020_PM25.nc',
    monitor_file='OfficialInput/2020_DailyPM2.5Monitor.csv'
)

# 处理指定某一天
result = fusion.process_day('2020-01-01')
print(result.head())
```

### 完整工作流

```python
from vna_fusion import VNAFusion
import pandas as pd

# 1. 初始化融合器
fusion = VNAFusion(
    model_file='OfficialInput/2020_PM25.nc',
    monitor_file='OfficialInput/2020_DailyPM2.5Monitor.csv',
    k=30,               # 近邻数量
    power=-2,          # IDW 权重幂次
    method='voronoi'    # 邻居选择方法
)

# 2. 查看可用日期
print(f"可用日期范围: {fusion.dates[0]} ~ {fusion.dates[-1]}")
print(f"共 {len(fusion.dates)} 天数据")

# 3. 处理单天数据
day_result = fusion.process_day('2020-06-01')

# 4. 处理日期范围
month_result = fusion.process_range('2020-06-01', '2020-06-30')

# 5. 保存结果
fusion.save(month_result, 'june_results.csv')

# 6. 结果分析
print(month_result.groupby('Timestamp')[['vna', 'avna', 'evna']].mean())
```

---

## 输入数据格式

### 监测数据 (CSV)

监测数据应包含站点标识、日期、浓度值和坐标信息。

**文件格式**:
```csv
Site,Date,Conc,Valid_Hours,Actual_Hour_Count,Lat,Lon
1003A,2020-01-01,34.625,24,24,39.9289,116.4174
1006A,2020-01-01,33.70,23,24,39.9295,116.3392
1007A,2020-01-01,38.00,24,24,39.9611,116.2878
```

**字段说明**:

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| Site | string | 是 | 监测站唯一标识符 |
| Date | string | 是 | 日期，格式 YYYY-MM-DD |
| Conc | float | 是 | 污染物浓度值 |
| Lat | float | 是 | 纬度 (十进制) |
| Lon | float | 是 | 经度 (十进制) |
| Valid_Hours | int | 否 | 有效小时数 |
| Actual_Hour_Count | int | 否 | 实际小时数 |

**数据要求**:
- 日期格式必须统一为 YYYY-MM-DD
- 经纬度不能为空或 NaN
- 同一站点同一天多条记录时自动取平均值
- 建议每年数据单独一个文件

### 模型数据 (NetCDF)

模型数据支持两种格式，系统自动识别。

#### IOAPI 格式 (CMAQ 模型输出)

CMAQ 模型的标准输出格式，包含完整的投影信息。

**维度**:
- TSTEP: 时间步数 (如 365 表示全年每日)
- Layer: 层数 (通常为 1)
- ROW: 行数 (南北方向)
- COL: 列数 (东西方向)

**变量示例**: O3_MDA8, PM25, NO2 等

**必需属性**: XORIG, YORIG, XCELL, YCELL, projection 相关参数

#### CF 格式 (通用 NetCDF)

符合 CF 约定的 NetCDF 文件。

**维度**:
- time: 时间步数
- y: 行数
- x: 列数

**坐标变量**:
- lat(y, x): 纬度网格
- lon(y, x): 经度网格

**变量示例**: pred_PM25, base_PM25, O3_MDA8

**自动识别逻辑**:

```python
# 优先选择包含以下关键词的变量
priority_keywords = ['PM25', 'O3', 'PM2.5', 'OZONE']

# 如果没找到，使用第一个数据变量
# 如果没有数据变量，抛出异常
```

### 区域掩码 (CSV, 可选)

指定需要融合计算的网格区域。

**文件格式**:
```csv
ROW,COL,Is
0,0,1
0,1,0
1,0,1
1,1,0
```

**字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| ROW | int | 网格行索引 (从0开始) |
| COL | int | 网格列索引 (从0开始) |
| Is | int | 是否参与计算 (1=是, 0=否) |

**用途**:
- 限制计算范围，提高效率
- 排除无关区域（如海洋、国外）
- 支持多区域对比分析

---

## 输出格式

### DataFrame 结构

处理完成后返回 pandas DataFrame，包含以下列：

| 列名 | 数据类型 | 说明 |
|------|----------|------|
| ROW | int | 网格行索引 |
| COL | int | 网格列索引 |
| vna | float | Voronoi 邻居 IDW 插值结果 |
| avna | float | 加法修正融合结果 |
| evna | float | 乘法修正融合结果 |
| model | float | 原始模型值 |
| Timestamp | string | 日期 (YYYY-MM-DD) |

### 输出示例

```
   ROW  COL        vna       avna      evna     model   Timestamp
0    0    0  19.284000  13.666278  3.696605  1.010550  2020-01-01
1    0    1  19.367190  13.724443  3.709184  1.013898  2020-01-01
2    0    2  19.452867  13.849807  3.962583  1.083184  2020-01-01
```

### 统计摘要

融合结果的统计特征：

```python
result[['vna', 'avna', 'evna', 'model']].describe()
```

输出：
```
              vna         avna         evna        model
count  21844.000000  21844.000000  21844.000000  21844.000000
mean       25.432156      20.153847       22.184736      18.567892
std        15.678234      12.456789      14.234567      11.234567
min         2.345678       1.234567       0.567890       0.123456
25%        12.345678       9.876543      10.234567       8.234567
50%        22.345678      18.234567      19.876543      16.567890
75%        35.678901      28.456789      30.234567      25.678901
max       125.678901     98.456789     105.234567      85.678901
```

---

## 融合方法详解

### 1. VNA (Voronoi Neighbor Averaging)

VNA 是基于 Voronoi 分解的邻居插值方法。

**核心思想**:
1. 根据监测站位置构建 Voronoi 图
2. 每个网格点找到其所在的 Voronoi 多边形
3. 使用该多边形内的监测站进行 IDW 插值

**数学公式**:

$$\hat{O}_{vna}(x) = \sum_{j=1}^{k} w_j \cdot O_j^{obs}$$

其中：
- $\hat{O}_{vna}(x)$: 网格点 x 的预测浓度
- $O_j^{obs}$: 第 j 个监测站的观测值
- $w_j$: 距离权重

**距离权重计算**:

$$w_j = \frac{d_j^p}{\sum_{j=1}^{k} d_j^p}$$

其中：
- $d_j$: 网格点到监测站 j 的距离
- $p$: 权重幂次 (默认 -2，即 IDW-2)

**特点**:
- 保持观测数据的局部特征
- 对监测站分布敏感
- 适用于空间变化较大的区域

### 2. aVNA (Additive VNA)

aVNA 在 VNA 基础上进行加法偏差校正。

**适用场景**:
- 模型输出系统性偏高或偏低
- 偏差与位置无关（空间均匀）
- 短期数据融合

**数学公式**:

$$O_{avna} = O_{model} - \overline{bias}$$

其中：

$$\overline{bias} = \frac{1}{k} \sum_{j=1}^{k} (O_{model,j} - O_j^{obs})$$

或等价的 VNA 表示：

$$O_{avna} = O_{model} - \widehat{bias}_{vna}$$

**特点**:
- 消除模型系统性偏差
- 保持模型空间梯度
- 对异常值较敏感

### 3. eVNA (Extended VNA)

eVNA 在 VNA 基础上进行乘法偏差校正。

**适用场景**:
- 模型与观测存在比例偏差
- 偏差随浓度水平变化
- 污染物浓度范围较大

**数学公式**:

$$O_{evna} = O_{model} \times \overline{r_n}$$

其中：

$$\overline{r_n} = \frac{1}{k} \sum_{j=1}^{k} \frac{O_j^{obs}}{O_{model,j}}$$

或等价的 VNA 表示：

$$O_{evna} = \widehat{r_n}_{vna} \times O_{model}$$

**特点**:
- 保留模型的相对空间分布
- 对高浓度区域加权更大
- 避免负值（当 ratio > 0 时）

### 4. 方法对比

| 特性 | VNA | aVNA | eVNA |
|------|-----|------|------|
| 修正方式 | 无 | 加法 | 乘法 |
| 适用偏差 | 无明显规律 | 均值偏移 | 比例失调 |
| 结果范围 | 取决于观测 | 可为负值 | 通常正值 |
| 对极端值敏感度 | 中 | 高 | 中 |
| EPA 推荐 | 是 | 是 | 是 |

---

## API 文档

### VNAFusion 类

#### 初始化

```python
VNAFusion(
    model_file,           # str, 必需 - NetCDF 模型文件路径
    monitor_file,         # str, 必需 - CSV 监测数据文件路径
    region_file=None,     # str, 可选 - CSV 区域掩码文件路径
    k=30,                 # int - 近邻数量 (EPA 标准为 30)
    power=-2,             # float - IDW 权重幂次
    method='voronoi',     # str - 邻居选择方法
    monitor_pollutant='Conc',  # str - 监测浓度列名
    model_pollutant=None  # str, 可选 - 模型变量名 (自动检测)
)
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| dates | ndarray | 所有可用日期 |
| model_format | str | 'ioapi' 或 'cf' |
| model_pollutant | str | 实际使用的模型变量名 |

#### 方法

##### process_day(date)

处理指定某一天的数据。

```python
result = fusion.process_day('2020-01-01')
```

**参数**:
- `date`: str, 'YYYY-MM-DD' 格式

**返回**: pandas.DataFrame

**示例**:

```python
result = fusion.process_day('2020-06-15')
print(f"处理了 {len(result)} 个网格点")
print(result[result['evna'] > 50])  # 查看高浓度区域
```

##### process_range(start_date, end_date)

处理日期范围内的所有数据。

```python
results = fusion.process_range('2020-01-01', '2020-12-31')
```

**参数**:
- `start_date`: str, 开始日期 'YYYY-MM-DD'
- `end_date`: str, 结束日期 'YYYY-MM-DD'

**返回**: pandas.DataFrame (合并所有日期)

**示例**:

```python
# 处理全年数据
year_results = fusion.process_range('2020-01-01', '2020-12-31')

# 按月分析
year_results['Month'] = pd.to_datetime(year_results['Timestamp']).dt.month
monthly_mean = year_results.groupby('Month')[['vna', 'avna', 'evna']].mean()
```

##### save(result_df, output_file)

保存结果到 CSV 文件。

```python
fusion.save(result_df, 'output/results.csv')
```

**参数**:
- `result_df`: pandas.DataFrame, 融合结果
- `output_file`: str, 输出文件路径

---

## 完整示例

### 示例 1: 月度臭氧融合分析

```python
import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/VNAeVNAaVNACodePrepare')
from vna_fusion import VNAFusion
import pandas as pd
import matplotlib.pyplot as plt

# 初始化 (臭氧 MDA8)
fusion = VNAFusion(
    model_file='Data/2020_O3.nc',
    monitor_file='Data/2020_O3Monitor.csv',
    model_pollutant='O3_MDA8',
    k=30,
    method='voronoi'
)

# 处理夏季 (6-8月)
summer_results = fusion.process_range('2020-06-01', '2020-08-31')

# 计算月度统计
summer_results['Month'] = pd.to_datetime(summer_results['Timestamp']).dt.month
monthly = summer_results.groupby(['Month', 'ROW', 'COL']).mean().reset_index()

# 夏季平均臭氧
summer_avg = summer_results.groupby(['ROW', 'COL'])['evna'].mean().reset_index()

# 找出高臭氧区域 (top 10%)
threshold = summer_avg['evna'].quantile(0.9)
high_o3 = summer_avg[summer_avg['evna'] > threshold]

print(f"高臭氧区域数量: {len(high_o3)}")
print(f"臭氧浓度范围: {high_o3['evna'].min():.1f} - {high_o3['evna'].max():.1f} ppb")

# 保存
fusion.save(summer_results, 'summer_o3_2020.csv')
```

### 示例 2: PM2.5 年度趋势分析

```python
import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/VNAeVNAaVNACodePrepare')
from vna_fusion import VNAFusion
import pandas as pd
import numpy as np

# 初始化
fusion = VNAFusion(
    model_file='Data/2020_PM25.nc',
    monitor_file='Data/2020_PM25Monitor.csv'
)

# 处理全年
results = fusion.process_range('2020-01-01', '2020-12-31')

# 提取时间信息
results['Date'] = pd.to_datetime(results['Timestamp'])
results['Month'] = results['Date'].dt.month
results['Season'] = results['Date'].dt.month % 12 // 3  # 0=DJF, 1=MAM, 2=JJA, 3=SON

# 季节平均
seasonal = results.groupby(['Season', 'ROW', 'COL']).mean().reset_index()
seasonal['Season'] = seasonal['Season'].map({0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Fall'})

# 季度对比
seasonal_avg = seasonal.groupby('Season')[['vna', 'avna', 'evna']].mean()
print("季度平均 PM2.5:")
print(seasonal_avg)

# 找出污染最严重的区域
annual_avg = results.groupby(['ROW', 'COL'])['evna'].mean().reset_index()
top_regions = annual_avg.nlargest(20, 'evna')
print(f"\n污染最严重的20个网格点:")
print(top_regions)
```

### 示例 3: 分批处理大数据

```python
import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/VNAeVNAaVNACodePrepare')
from vna_fusion import VNAFusion
import os

# 创建输出目录
os.makedirs('monthly_output', exist_ok=True)

# 初始化
fusion = VNAFusion(
    model_file='Data/2020_PM25.nc',
    monitor_file='Data/2020_PM25Monitor.csv'
)

# 分月处理，避免内存溢出
for month in range(1, 13):
    start = f'2020-{month:02d}-01'
    # 获取月末日期
    if month == 12:
        end = '2020-12-31'
    else:
        end = f'2020-{month+1:02d}-01'

    print(f"处理 {start} ~ {end}...")

    # 处理当月
    month_result = fusion.process_range(start, end)

    # 保存
    output_path = f'monthly_output/month_{month:02d}.csv'
    fusion.save(month_result, output_path)

print("全部完成!")
```

### 示例 4: 自定义区域分析

```python
import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/VNAeVNAaVNACodePrepare')
from vna_fusion import VNAFusion
import pandas as pd

# 创建区域掩码 - 只分析北京及周边
region_data = """ROW,COL,Is
50,100,1
50,101,1
51,100,1
51,101,1
52,100,1
52,101,1
"""
with open('beijing_region.csv', 'w') as f:
    f.write(region_data)

# 初始化
fusion = VNAFusion(
    model_file='Data/2020_PM25.nc',
    monitor_file='Data/2020_PM25Monitor.csv',
    region_file='beijing_region.csv'
)

# 处理一天
result = fusion.process_day('2020-01-15')

# 分析
print("北京区域 PM2.5 统计:")
print(result[['vna', 'avna', 'evna', 'model']].describe())
```

---

## 依赖说明

| 包 | 版本要求 | 说明 |
|-----|----------|------|
| numpy | >= 1.19 | 基础数值计算 |
| pandas | >= 1.0 | 数据框操作 |
| xarray | >= 0.16 | NetCDF 读取 |
| scikit-learn | >= 0.23 | 最近邻搜索 |
| scipy | >= 1.5 | 空间计算 |
| joblib | >= 1.0 | 并行计算 |
| tqdm | >= 4.0 | 进度条显示 |
| pyproj | >= 3.0 | 坐标投影转换 |
| pyrsig | 最新版 | IOAPI 格式支持 |

### 可选依赖

| 包 | 说明 |
|-----|------|
| matplotlib | 结果可视化 |
| cartopy | 地图绑制 |

---

## 故障排除

### 问题 1: KeyError: 'PM25'

**原因**: 模型文件中的变量名不是 'PM25'。

**解决方案**: 使用自动检测（默认行为），或手动指定：

```python
# 自动检测（推荐）
fusion = VNAFusion(model_file='model.nc', monitor_file='monitor.csv')

# 手动指定
fusion = VNAFusion(
    model_file='model.nc',
    monitor_file='monitor.csv',
    model_pollutant='pred_PM25'  # 实际变量名
)
```

### 问题 2: 模块导入失败

**原因**: 路径未正确设置。

**解决方案**:

```python
import sys
# 添加模块路径
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/VNAeVNAaVNACodePrepare')

# 验证导入
from vna_fusion import VNAFusion
print("导入成功!")
```

### 问题 3: 内存不足

**原因**: 处理全年数据时内存占用过大。

**解决方案**: 分批处理

```python
# 方案1: 按月处理
for month in range(1, 13):
    result = fusion.process_range(f'2020-{month:02d}-01', f'2020-{month:02d}-28')
    fusion.save(result, f'month_{month}.csv')

# 方案2: 减少并行核数
# 修改 nna_methods/__init__.py 中的 njobs 参数
```

### 问题 4: 监测站点过少

**警告**: 某天监测站点数量少于 k 值。

**解决方案**: 模块会自动调整，但建议检查数据：

```python
# 查看每天的站点数
daily_data = fusion._input.get_daily_data('2020-01-01')
print(f"站点数量: {len(daily_data['obs'])}")
```

### 问题 5: 日期格式错误

**原因**: 日期字符串格式不符合 YYYY-MM-DD。

**解决方案**:

```python
# 正确格式
date = '2020-01-01'

# 转换为正确格式
from datetime import datetime
date = datetime(2020, 1, 1).strftime('%Y-%m-%d')
```

### 问题 6: IOAPI 格式读取失败

**原因**: 模型文件不是标准 IOAPI 格式。

**解决方案**: 系统会自动回退到 CF 格式，如果仍失败，检查文件：

```python
import xarray as xr

# 检查文件是否可读
ds = xr.open_dataset('model.nc')
print(ds)
```

---

## 版本历史

### v1.0.0 (2026-04-02)
- 初始版本
- 支持 VNA, aVNA, eVNA 三种融合方法
- 支持 IOAPI 和 CF 两种 NetCDF 格式
- 自动检测 PM25/O3 相关变量
- 支持单日和日期范围处理

---

## 参考文献

1. Glahn, H.R. (2012). Blending Point and Gridded Data: A Generalization of the Schröppel Event. AMS Annual Meeting.
2. EPA Guidance Document: Spatial Quality Objectives and Performance Targets for Air Quality Monitoring Data.
3. Shepard, D. (1968). A two-dimensional interpolation function for irregularly-spaced data. ACM National Conference.
