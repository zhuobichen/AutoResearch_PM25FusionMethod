# 创新方法指令

## 方法名称
Multi-Kernel GPR PolyRK (MKGPR-RK) - 多核高斯过程残差多项式克里金

## 创新点
在 PolyRK 的二次多项式 OLS 全局校正 + 局部 GPR 克里金混合架构基础上，使用多核（multi-kernel）GPR 替代单一 RBF 核 GPR。单一核函数难以同时捕捉空间相关性的多尺度特性（如短程城市尺度变异和长程区域尺度趋势），多核 GPR 通过叠加多个具有不同长度尺度的核函数，可以更好地描述残差的空间结构。

## 背景问题

### 单一 RBF 核的局限性
PolyRK 使用的 RBF 核：
$$
k_{RBF}(s_i, s_j) = \sigma^2 \exp\left(-\frac{||s_i - s_j||^2}{2\ell^2}\right)
$$
单一长度尺度 $\ell$ 假设空间相关性在所有尺度上均匀衰减。

### 实际空间相关性的多尺度特性
大气污染数据的空间变异通常包含：

1. **短程变异（< 20 km）**：城市热岛、局地源排放、建筑群影响
2. **中程变异（20-100 km）**：区域背景浓度梯度
3. **长程变异（> 100 km）**：大尺度气象传输、区域污染带

单一 RBF 核只能捕捉一个典型尺度，多核方法可以同时覆盖多个尺度。

### 多核融合的物理动机
不同长度尺度的核函数对应不同物理过程：
- 短程核（RBF with small ℓ）：局地源影响、高分辨率网格变异
- 长程核（RBF with large ℓ）：区域背景、大尺度趋势
- WhiteKernel：监测噪声

## 核心公式

### 第一步：全局多项式校正（与 PolyRK 相同）
$$
M_{cal}(s) = \alpha_0 + \alpha_1 M(s) + \alpha_2 M(s)^2
$$

### 第二步：残差计算
$$
R_i = O_i - M_{cal}(s_i)
$$

### 第三步：多核 GPR 建模
**多核协方差函数**：
$$
k_{multi}(s_i, s_j) = \sum_{k=1}^{K} \sigma^2_k \exp\left(-\frac{||s_i - s_j||^2}{2\ell_k^2}\right) + \sigma^2_{noise} \delta_{ij}
$$

其中：
- $k=1$: 短程核（ℓ₁ ≈ 5-15 km，捕捉局地变异）
- $k=2$: 中程核（ℓ₂ ≈ 30-50 km，捕捉区域梯度）
- $k=3$: 长程核（ℓ₃ ≈ 100-200 km，捕捉大尺度趋势）
- $\sigma^2_k$: 各核的方差权重（通过边缘似然优化）

**简化版三核配置**：
$$
k_{multi} = \sigma^2_1 \cdot RBF(\ell_1) + \sigma^2_2 \cdot RBF(\ell_2) + WhiteKernel(\sigma^2_{noise})
$$

### 第四步：融合结果
$$
P_{MKGPRK}(s) = M_{cal}(s) + \hat{R}(s)
$$

## 关键步骤

### Step 1: 多核定义与初始化
```
输入: X_train (lon, lat)
处理:
  1. 定义多核:
     kernel = (
         ConstantKernel(10.0) * RBF(length_scale=10.0) +   # 短程
         ConstantKernel(10.0) * RBF(length_scale=40.0) +   # 中程
         WhiteKernel(noise_level=1.0)                      # 噪声
     )
  2. 设置长度尺度边界:
     - ℓ1: (1.0, 20.0) km  短程
     - ℓ2: (20.0, 100.0) km 中程
     - ℓ3: 不需要（WhiteKernel 无长度尺度）
输出: kernel
```

### Step 2: GPR 拟合（与 PolyRK 相同接口）
```
输入: X_train, residual, kernel
处理:
  1. gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=3,  # 多核需要更多重启
        alpha=0.1,
        normalize_y=True
    )
  2. gpr.fit(X_train, residual)
  3. 记录优化后的核参数和边缘似然
输出: gpr 模型
```

### Step 3: 预测
```
输入: X_test, gpr
处理:
  1. R_pred = gpr.predict(X_test)
  2. 可选: R_std = gpr.predict(X_test, return_std=True)[1]
输出: R_pred
```

## 参数清单

| 参数符号 | 物理意义 | 取值范围 | 建议初值 |
|---------|---------|---------|---------|
| ℓ₁ | 短程核长度尺度 | 1-20 km | 10.0 |
| ℓ₂ | 中程核长度尺度 | 20-100 km | 40.0 |
| ℓ₃ | 长程核长度尺度 | 100-300 km | 150.0（可选） |
| σ²_1 | 短程核方差权重 | 1e-2-1e2 | 10.0 |
| σ²_2 | 中程核方差权重 | 1e-2-1e2 | 10.0 |
| σ²_noise | 噪声方差 | 1e-5-1e1 | 1.0 |
| n_restarts | GPR 优化重启 | 2-5 | 3 |
| GPR alpha | GPR 正则化 | 0.01-1.0 | 0.1 |

## 与 PolyRK 的差异

| 方面 | PolyRK | MKGPR-RK |
|------|--------|----------|
| GPR 核函数 | 单一 RBF + WhiteKernel | 多 RBF 叠加 + WhiteKernel |
| 空间相关性 | 单尺度（典型 ℓ ≈ 15 km） | 多尺度（ℓ₁≈10 + ℓ₂≈40） |
| 短程变异捕捉 | 可能被平均化 | 独立短程核专门捕捉 |
| 长程趋势捕捉 | 依赖 GPR 全局插值 | 独立长程核直接建模 |
| 计算复杂度 | O(n²) | O(n²)（核数增加轻微影响） |
| 边缘似然优化 | 单核优化 | 多核联合优化（更多局部极小） |

**核心差异**：PolyRK 的 GPR 使用单一 RBF 核，隐含假设空间相关性在单一典型尺度上衰减；MKGPR-RK 使用多核叠加，允许同时存在多个空间相关尺度，更真实地反映大气污染的多尺度空间结构。

## 预期效果
- R² >= 0.855（比 PolyRK 的 0.8519 提升约 0.003）
- MAE <= 7.0, RMSE <= 10.9
- 特别在空间变异复杂（多尺度源影响）的数据集上优势更明显

## 方法指纹
MD5: `multi_kernel_gpr_polyrk_v1`

## 禁止依赖
- 不依赖气象数据（风速、温度、稳定度等）
- 不依赖时间数据（日期、季节等）
- 仅依赖 CMAQ 格点数据 + 监测站点坐标

## sklearn 实现参考
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def mkgpr_rk_fit(X_train, residual):
    """MKGPR-RK 训练"""
    # 多核定义: 短程 + 中程 + 噪声
    kernel = (
        ConstantKernel(10.0, (1e-2, 1e3)) *
        RBF(length_scale=10.0, length_scale_bounds=(1.0, 20.0)) +
        ConstantKernel(10.0, (1e-2, 1e3)) *
        RBF(length_scale=40.0, length_scale_bounds=(20.0, 100.0)) +
        WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=3,  # 多核需要更多重启
        alpha=0.1,
        normalize_y=True
    )
    gpr.fit(X_train, residual)

    # 打印优化后的核参数
    print(f"Optimized kernel: {gpr.kernel_}")

    return gpr

def mkgpr_rk_predict(X_test, gpr):
    """MKGPR-RK 预测"""
    R_pred = gpr.predict(X_test)
    return R_pred
```

## 扩展：五核版本
对于更大空间尺度的数据，可扩展至五核：
```
kernel = (
    ConstantKernel(5.0) * RBF(length_scale=5.0) +    # 超短程（站点局部）
    ConstantKernel(10.0) * RBF(length_scale=15.0) +  # 短程（城市尺度）
    ConstantKernel(10.0) * RBF(length_scale=50.0) +   # 中程（区域尺度）
    ConstantKernel(10.0) * RBF(length_scale=150.0) + # 长程（大尺度）
    WhiteKernel(noise_level=1.0)                      # 噪声
)
```

## 优化建议
1. **核函数选择**：优先使用 RBF（Mattern 核需要特殊函数，计算更慢）
2. **长度尺度初始化**：基于数据覆盖范围的经验值
3. **重启次数**：多核优化更容易陷入局部最优，建议 n_restarts >= 3
4. **提前终止**：如果边缘似然在连续两次重启间变化 < 1%，可提前终止
