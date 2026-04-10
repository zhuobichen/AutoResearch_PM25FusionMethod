# STRK 失败诊断报告

## 方法信息
- **方法名**: STRK (Spatio-Temporal Residual Co-Kriging)
- **测试日期**: 2026-04-09
- **测试结果**: R²=0.3902, MAE=15.82, RMSE=22.42

## 诊断结论

### 根本原因：时间建模失效

STRK 核心创新依赖时间自回归模型捕捉日变化规律。

**问题**：
1. `cross_validate()` 只传入单日数据 `selected_days=['2020-01-01']`
2. `fit_temporal_model()` 检测到 `monitor_df=None`，设置 `ar_coef=0.0`
3. 时间残差预测 `R_temp_pred = 0.0 * R_spatial_pred = 0`
4. 融合退化为：`Z* = Z_RK + θ1*R_systematic + θ3*R_st`

### 关键代码问题

```python
# fit_temporal_model() 第306-308行
if monitor_df is None or dates is None:
    self.ar_coef = 0.0
    return
```

单日验证场景下，时间建模完全失效。

### 修复建议

**方案A - 简化版STRK（推荐）**：
去掉时间残差项，仅保留空间残差克里金

**方案B - 多日联合验证**：
修改验证流程，传入连续多日数据使时间建模生效

## 结论

R²=0.3902 低于 0.70 阈值，因时间建模需要多日数据但验证只用单日。
