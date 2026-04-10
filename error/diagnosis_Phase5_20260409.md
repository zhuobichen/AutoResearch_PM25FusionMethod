# Phase 5 失败诊断报告
生成日期: 2026-04-09

## 1. 测试结果确认

| 方法 | R² | MAE | RMSE | 状态 |
|------|-----|------|------|------|
| SQDM | 0.7887 | 8.00 | 13.20 | 低于基准 |
| LBGPR | 0.6506 | 11.29 | 16.97 | 严重低于基准 |
| GARK | 0.6403 | 11.63 | 17.22 | 严重低于基准 |
| **PolyRK_Poly (基准)** | **0.8519** | **7.09** | **11.05** | **成功** |

---

## 2. 核心问题诊断

### 2.1 SQDM (R²=0.7887) 分析

**问题**: 使用简单IDW加权平均，而非真正的空间克里金插值

**证据** - `SQDM.py` 第137-203行 `sqdm_predict` 函数:
```python
# 空间权重
spatial_weights = 1.0 / (dists_k ** alpha_spatial)
# CMAQ相似性权重
cmaq_weights = np.exp(-gamma * cmaq_diff)
# 综合权重
combined_weights = spatial_weights * cmaq_weights
pred_values[i] = np.sum(combined_weights * r_k)  # 纯加权平均
```

**根本原因**: 
- SQDM本质上是**IDW（反距离加权）**变体
- 没有使用GPR进行空间相关性建模
- 无法捕捉空间变差函数的结构

**对比PolyRK**: PolyRK使用GPR克里金：
```python
# PolyRK.py 第137-144行
gpr_poly.fit(X_train, residual_poly)  # 训练GPR
gpr_poly_pred, _ = gpr_poly.predict(X_test, return_std=True)  # GPR预测
```

---

### 2.2 LBGPR (R²=0.6506) 分析

**问题1**: 函数名是LBGPR，但实际实现是纯IDW

**证据** - `LBGPR.py` 第117-171行 `lbgpr_predict_efficient` 函数:
```python
# 使用局部带宽的高斯权重
bw = local_bw[i]
weights = np.exp(-0.5 * (dists_k / bw)**2)
weights = weights / weights.sum()
# 使用加权平均作为预测
pred_values[i] = np.sum(weights * y_k)  # 没有GPR调用！
```

**问题2**: 没有进行真正的GPR训练和预测

LBGPR代码中虽然有GPR对比代码（第246-250行），但实际报告的LBGPR结果是IDW版本。

---

### 2.3 GARK (R²=0.6403) 分析

**问题1**: 各向异性权重计算存在bug

**证据** - `GARK.py` 第201-203行:
```python
# 高斯相关函数权重 - 长度尺度固定为 a_min/3.0
weights = np.exp(-0.5 * (dists_k / (a_min / 3.0))**2)
weights = weights / weights.sum()
```

**问题2**: 没有使用GPR进行克里金

- 直接使用高斯权重函数计算加权平均
- 没有训练GPR模型
- 各向异性参数 (a_min=8.0, a_max=20.0, alpha=2.0) 可能是次优的

---

## 3. 失败根因总结

| 方法 | 失败根因 | 建议改进方向 |
|------|----------|-------------|
| SQDM | 纯IDW，无GPR建模 | 改用GPR克里金替代IDW |
| LBGPR | 名不副实，实际是IDW | 实现真正的局部带宽GPR |
| GARK | 各向异性权重有bug，无GPR | 修复权重计算，使用各向异性GPR核 |

### 关键差异：为什么PolyRK成功？

PolyRK成功的关键在于：
1. **OLS多项式回归**: O = a + b*M + c*M² 捕捉非线性偏差
2. **GPR克里金**: 对残差进行高斯过程回归，捕捉空间相关性
3. **完整的概率预测**: GPR提供预测均值和方差

---

## 4. 改进建议

### 4.1 SQDM改进
```python
# 当前: 纯IDW
pred = sum(w_i * residual_i)

# 建议: 使用GPR
gpr = GaussianProcessRegressor(kernel=kernel)
gpr.fit(X_train, residual_train)
gpr_pred = gpr.predict(X_test)
```

### 4.2 LBGPR改进
```python
# 当前: 加权IDW
pred = sum(w_i * y_i)

# 建议: 真正的局部带宽GPR
# 对每个预测点，使用局部带宽训练独立的GPR
```

### 4.3 GARK改进
```python
# 当前: 各向异性距离 + 简单高斯权重
dists = anisotropic_distance(...)
weights = exp(-0.5 * (dists / (a_min/3.0))**2)

# 建议: 使用各向异性GPR核
from sklearn.gaussian_process.kernels import AnisotropicGaussianProcess
kernel = AnisotropicGaussianProcess(a_min, a_max)
gpr = GaussianProcessRegressor(kernel=kernel)
```

---

## 5. 结论

**主要发现**: SQDM/LBGPR/GARK 失败的核心原因是**没有正确使用GPR进行空间克里金插值**。

PolyRK的成功在于：
1. 使用OLS多项式进行全局偏差校正
2. 使用GPR对残差进行克里金插值

这三种方法（尤其是LBGPR和GARK）都是**局部方法**，在十折验证的小样本情况下表现不佳。而PolyRK的**全局多项式+局部克里金**混合方法更好地平衡了全局趋势和局部变异。

---

**诊断人**: Claude Code Agent
**日期**: 2026-04-09
