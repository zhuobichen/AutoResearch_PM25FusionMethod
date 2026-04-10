# 创新方法指令

## 方法名称
SPIN-Kr：图核时空克里金法 (Spatiotemporal Physics-Informed Graph Kernel Kriging)

## 方法来源
论文：Physics-Guided Inductive Spatiotemporal Kriging for PM2.5 with Satellite Gradient Constraints (Wang et al., 2025)
arXiv: 2511.16013v1

## 创新核心
将AOD卫星梯度作为损失函数约束（非直接输入），结合图核建模时空邻域关系，实现归纳式时空克里金插值。

## 输入数据
- 监测站坐标：shape (n, 2)
- CMAQ数据：shape (lat, lon, time)
- AOD数据：shape (lat_aod, lon_aod, time) - 可缺失
- 监测PM2.5：shape (n, time)

## 输出数据
- 融合结果：shape (lat, lon, time)

## 核心公式

### 图核定义
$$
k_{graph}(s_i, s_j) = \exp\left(-\frac{d_{graph}(s_i, s_j)}{\lambda}\right)
$$

**图距离计算：**
$$
d_{graph}(s_i, s_j) = \alpha \cdot d_{euclidean}(s_i, s_j) + (1-\alpha) \cdot d_{wind}(s_i, s_j)
$$
其中 $d_{wind}$ 是沿风向的距离（考虑污染传输方向）

### AOD梯度约束损失项
$$
L_{gradient} = \frac{1}{N_{grid}}\sum_{(s,t)}\left|\nabla \cdot (CMAQ \cdot v_{wind}) - \frac{\partial AOD}{\partial t}\right|^2
$$
将AOD时间变化作为污染平流发散的代理约束

### 克里金权重求解
$$
w = K^{-1} \cdot k
$$
其中 $K_{ij} = k_{graph}(s_i, s_j) + \sigma_n^2 \delta_{ij}$（带噪声的图核矩阵）

### 融合预测
$$
P_{SPIN}(s_0, t) = \sum_{i=1}^{n} w_i(t) \cdot O(s_i, t) + \beta \cdot CMAQ(s_0, t)
$$
其中 $w_i(t)$ 是时变克里金权重，$\beta$ 是CMAQ缩放因子

## 关键步骤

### Step 1: 构建空间图结构
```
输入: X_stations (lon, lat), wind_direction
处理:
  1. 计算欧氏距离矩阵 D_euclidean
  2. 沿风向计算传输距离 D_wind
  3. 融合: D_graph = alpha * D_euclidean + (1-alpha) * D_wind
  4. 构建图核矩阵: K_ij = exp(-D_graph / lambda)
输出: 图核矩阵 K
```

### Step 2: 时变权重学习
```
输入: K, timestamps
处理:
  1. 对每个时间步 t:
     - 提取 K(t) 子矩阵
     - 添加时间正则化: K_reg = K + gamma * I
     - 求解 w(t) = K_reg^{-1} * k(t, s_0)
  2. 权重平滑: 对 w(t) 应用时间滑动平均
输出: 时变权重 w(t)
```

### Step 3: AOD梯度约束优化
```
输入: CMAQ_grid, AOD_grid, wind_field
处理:
  1. 计算 CMAQ 的平流散度: div(CMAQ * v_wind)
  2. 计算 AOD 时间导数: dAOD/dt
  3. 最小化梯度损失: min_beta ||div(CMAQ*v) - dAOD/dt||^2
  4. 得到最优缩放因子 beta
输出: beta
```

### Step 4: 克里金预测
```
输入: O_stations, w(t), beta, CMAQ_grid
处理:
  P = sum_i w_i(t) * O_i + beta * CMAQ_grid
输出: 融合结果 P
```

## 【创新点】

1. **无权重学习**：克里金权重由图核确定（非数据驱动权重），beta由物理约束确定
2. **AOD作为梯度约束而非输入**：避免AOD缺失问题，仅用其时间梯度作为软约束
3. **图核建模传输距离**：沿风向的图距离比欧氏距离更能捕捉污染传输相关性
4. **归纳式学习**：图核可泛化到未观测位置

## 创新判定
- 是否使用权重学习（Ridge/Lasso/线性回归）？否
- 是否有物理可解释性？是（风导向图核+平流约束）
- 是否保留：创新成立

## 创新优势
- 相比普通ResidualKriging：图核考虑了风向，比各向同性核更物理
- 相比PDEICNN：不需要完整PDE约束，仅用AOD梯度作为软约束，更灵活
- 预期R²提升 >= 0.015

## 风险假设
- AOD缺失时梯度约束失效，但整体克里金仍可工作
- 图核参数(alpha, lambda)需物理初始化
- 验证计划：对比各向同性图核与风导向图核的CV-RMSE

## 方法指纹
MD5: `spin_graph_kernel_kriging_v1`

## 参数清单

| 参数符号 | 物理意义 | 取值范围 | 建议初值 |
|---------|---------|---------|---------|
| alpha | 欧氏-风向距离权重 | 0.0-1.0 | 0.5 |
| lambda | 图核长度尺度 (km) | 10-100 | 50.0 |
| gamma | 时间正则化 | 0.01-1.0 | 0.1 |
| beta | CMAQ缩放因子 | 0.8-1.2 | 1.0 |

## 输入输出格式
- 输入：监测数据CSV + CMAQ netCDF + AOD netCDF + 风场数据
- 输出：融合网格PM2.5
- 支持时间序列验证
