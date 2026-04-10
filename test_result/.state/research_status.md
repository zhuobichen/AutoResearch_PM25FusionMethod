# 研究状态报告

更新时间：2026-04-10

## 快照状态

| 字段 | 值 |
|------|-----|
| 当前快照 | round_0 |
| 快照目录 | test_result/snapshots/ |
| 运行模式 | 初始化（待第一次运行） |

## 快照使用说明

```
# 查看当前状态
python test_result/snapshot_manager.py

# 创建新快照（运行前）
manager = SnapshotManager(root)
manager.create_snapshot(round_num=1, note="开始第一次运行")

# 运行结束后保存结果
manager.update_best_method("RK-Poly", {"R2": 0.8519})
manager.update_note("预实验完成，效果不理想，需调整")

# 下次运行从 round_1 恢复
manager.restore_snapshot(1)
```

## 当前最佳方案

| 指标 | 值 |
|------|-----|
| 方法名 | 待验证 |
| R² | 需测试 |
| 类别 | 创新方法（排除Stacking类） |

## 基准方法

| 方法 | R² | RMSE | MAE | MB |
|------|-----|------|-----|-----|
| CMAQ | -0.0376 | 29.25 | 20.47 | -3.24 |
| VNA | 0.7996 | 12.86 | 7.75 | 0.76 |
| aVNA | 0.7941 | 13.03 | 8.10 | 0.10 |
| eVNA | 0.8100 | 12.52 | 7.99 | 0.08 |

## 去重追踪

| 类型 | 已记录数 | 说明 |
|------|----------|------|
| 已下载论文 | 0 | dedup_key列表 |
| 已分析方法 | 0 | 方法名列表 |
| 已有指纹 | 0 | 方法指纹MD5 |

## 目录结构状态

| 目录 | 状态 |
|------|------|
| .state/ | ✅ |
| snapshots/ | ✅ round_0已创建 |
| 基准方法/ | ✅ |
| 复现方法/ | ✅ |
| 创新方法/ | ✅ |
| 历史最佳方案/ | ✅ |
| 历史/ | ✅ |
| INVENTORY.md | ✅ |
| snapshot_manager.py | ✅ |
