# MSAK 失败诊断报告

## 方法信息
- **方法名**: MSAK (Multi-Scale Stability-Adaptive Kriging)
- **测试日期**: 2026-04-09
- **测试结果**: R²=0.1251, MAE=18.73, RMSE=26.86

## 诊断结论

### 根本原因：气象数据缺失

MSAK 核心创新依赖 Pasquill-Gifford 稳定度等级 (PG=1-6)，该等级需要风速(WS)、辐射或云量数据计算。

**数据现状**：
- 监测数据列：`Site, Date, Conc, Valid_Hours, Actual_Hour_Count, Lat, Lon`
- **缺少 WS（风速）列**

### 问题传播链

1. `fit()` 调用 `compute_stability(ws_col='WS')` 但数据无 WS 列
2. 代码 fallback: `ws = np.full(len(df), 3.0)` 全局填充风速=3.0 m/s
3. 所有站点 PG 等级计算结果相同（风速3.0 → PG=C=3）
4. 稳定度自适应权重 `α(PG)` 失去意义，所有站点 α 值相同
5. 多尺度 GPR 模型退化为单一尺度 GPR

### 修复建议

**方案A - 简化版MSAK（推荐）**：
去掉稳定度依赖，直接使用固定相关长度的多尺度克里金

**方案B - 引入气象数据**：
从外部气象数据源获取站点风速数据

## 结论

R²=0.1251 远低于 0.70 阈值，因核心创新假设（稳定度自适应）与数据不匹配。
