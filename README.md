# PM2.5 CMAQ融合方法自动研究系统

> 基于Agent Spawn模式的全自动化文献筛选 → 方法设计 → 代码实现 → 测试验证 → 论文生成流水线

[![研究进度](https://img.shields.io/badge/研究进度-Round%202-blue.svg)](https://github.com/zhuobichen/AutoResearch_PM25FusionMethod)
[![文档状态](https://img.shields.io/badge/文档-v11-green.svg)](./PM2.5_CMAQ融合方法自动研究全流程文档_v11_agent_spawn.md)

---

## 项目简介

本项目实现了一个**全自动化的PM2.5 CMAQ融合方法研究系统**，旨在：

1. **自动复现**现有融合方法（VNA, eVNA, aVNA, Downscaler等）
2. **自动提出**新的融合方法并验证其创新性
3. **自动生成**学术论文（LaTeX → PDF）

### 核心特点

| 特点 | 说明 |
|------|------|
| **Agent Spawn模式** | 多角色并行协作，文献下载(3并行) → 分析 → 设计 → 实现 → 验证 |
| **多阶段验证** | 预实验(5天) → Stage1(1月) → Stage2(7月) → Stage3(12月) |
| **十折交叉验证** | 标准模式 + 特例模式，确保验证公正性 |
| **快照机制** | 支持中断恢复，每次运行生成快照存档 |
| **去重机制** | 方法指纹MD5去重，避免重复研究 |

---

## 系统架构

### 工作流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PM2.5 CMAQ融合方法自动研究全流程                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 0: 项目整理                                                      │
│  └─→ 扫描已有文件，生成INVENTORY.md                                     │
│                                                                         │
│  Phase X: 初始化检查                                                    │
│  └─→ 验证基准方法代码、指标完整性、十折划分表                             │
│                                                                         │
│  Phase 1: 并行文献下载 (3个Agent并行)                                   │
│  └─→ 论文搜索 → PDF下载 → paper_list.json                               │
│                                                                         │
│  Phase 2: 文献分析                                                      │
│  └─→ 论文PDF解析 → 方法提取 → MethodToSmart/文献分析员_*.md             │
│                                                                         │
│  Phase 3: 方案设计                                                      │
│  └─→ 复现方案 + 创新方案 → SmartToCode/创新方法指令/                   │
│                                                                         │
│  Phase 4: 代码实现                                                      │
│  └─→ 方法代码 → CodeWorkSpace/                                         │
│                                                                         │
│  Phase 5: 测试验证                                                      │
│  └─→ 十折验证 → R²/RMSE/MAE/MB → 创新判定                             │
│                   ↓                                                      │
│         ┌────────┴────────┐                                              │
│      通过           未通过                                               │
│         ↓              ↓                                                 │
│    论文写作        继续下一轮迭代                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 基准方法

| 方法 | 英文名 | 说明 |
|------|--------|------|
| **VNA** | Voronoi Neighbor Average | 空间插值基准 |
| **eVNA** | enhanced VNA | 乘法偏差校正 |
| **aVNA** | additive VNA | 加法偏差校正 |
| **Downscaler** | MCMC Downscaler | 降尺度方法 |

### 创新判定标准

三个指标必须**同时满足**：

| 指标 | 要求 | 说明 |
|------|------|------|
| R²提升 | ≥ 最优基准R² + 0.01 | 相比4个基准中R²最高的 |
| RMSE | ≤ 最优基准RMSE | 不差于基准 |
| \|MB\| | ≤ 最优基准\|MB\| | 偏差不增大 |

---

## 目录结构

```
Data_Fusion_AutoResearch/
├── PaperDownload/           # 论文PDF文件（按评分分类）
├── PaperDownloadMd/         # 论文清单、分析报告
├── LocalPaperLibrary/       # 本地原始论文库
├── MethodToSmart/           # 文献分析员输出
├── SmartToCode/            # 方案设计师输出
│   ├── 复现方法指令/
│   └── 创新方法指令/
├── Code/                    # 参考代码
│   └── Downscaler/          # Downscaler变体(Nystrom/Sparse/Woodbury)
├── CodeWorkSpace/           # 工作区
│   ├── 基准方法代码/        # VNA, eVNA, aVNA, Downscaler
│   ├── 复现方法代码/        # 已复现方法
│   └── 新融合方法代码/      # 创新方法
├── test_data/               # 测试数据
│   ├── fold_split_table.csv # 十折交叉验证划分
│   └── selected_days.txt    # 测试日期列表
├── test_result/             # 测试结果
│   ├── 基准方法/            # benchmark_summary.csv
│   ├── 创新方法/            # 各方法验证结果
│   ├── 历史最佳方案/         # 当前最优方案
│   └── snapshots/           # 快照存档
├── Innovation/              # 已确认创新方法
│   ├── success/            # 通过验证的创新
│   └── failed/             # 未通过的创新
├── paper_output/            # 论文输出
├── agents/                  # Agent模块
│   ├── spawn_executor.py   # Agent spawn执行器
│   ├── role_templates.py    # 角色prompt模板
│   └── ...
└── run_pipeline.py         # 工作流启动脚本
```

---

## 快速开始

### 环境要求

- Python 3.8+
- Git
- LaTeX (xelatex, biber)
- Zotero (可选，用于文献管理)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行完整流程

```bash
# 启动工作流
python run_pipeline.py

# 查看当前状态
python run_pipeline.py --status
```

### 阶段运行

```bash
# 仅下载文献
python -c "from agents import SpawnExecutor; SpawnExecutor('.').phase1_download()"

# 仅测试验证
python -c "from agents import SpawnExecutor; SpawnExecutor('.').phase5_test()"
```

---

## 核心概念

### 十折交叉验证

**标准模式**（适用于VNA/eVNA/aVNA/RK-Poly等）：
```
训练输入：9折监测站数据 + 9折站点对应的CMAQ网格值
预测目标：1折监测站所在的CMAQ网格坐标
评估：网格预测值 vs 1折站点真实值
```

**特例模式**（适用于Downscaler）：
```
训练输入：9折监测站数据 + 全网格CMAQ
预测目标：全网格（必须）
评估：提取验证站点位置后对比
```

详见 [十折交叉验证架构文档](./十折交叉验证架构文档.md)

### 多阶段验证

| 阶段 | 数据范围 | 目的 |
|------|----------|------|
| 预实验 | 2020-01-01 ~ 2020-01-05 (5天) | 快速筛选 |
| Stage 1 | 2020-01-01 ~ 2020-01-31 (1月) | 1月验证 |
| Stage 2 | 2020-07-01 ~ 2020-07-31 (7月) | 7月验证 |
| Stage 3 | 2020-12-01 ~ 2020-12-31 (12月) | 12月验证 |

### 方法指纹去重

每个方法通过以下信息生成MD5指纹：
- 核心公式
- 关键步骤
- 参数列表

指纹相同的方法被认为是重复研究。

---

## 测试结果

### 基准方法参考指标 (2020-01-01单天)

| 方法 | R² | RMSE | MAE | MB |
|------|-----|------|-----|-----|
| CMAQ | -0.038 | 29.25 | 20.47 | -3.24 |
| VNA | 0.800 | 12.86 | 7.75 | 0.76 |
| aVNA | 0.794 | 13.03 | 8.10 | 0.10 |
| eVNA | 0.810 | 12.52 | 7.99 | 0.08 |
| Downscaler | 0.806 | 12.64 | 8.19 | 1.85 |

### 创新方法阈值

各阶段的创新阈值基于该阶段**最优基准**计算：

```
最优基准 = 4个基准方法中R²最高的那一个
创新阈值 = 最优基准R² + 0.01
```

---

## 创新方法排除规则

以下类型的方法**不应作为创新**：

| 类型 | 示例 | 原因 |
|------|------|------|
| 加权集成 | SuperStackingEnsemble | 权重迁移性差 |
| 无物理意义 | 纯神经网络黑盒 | 审稿人挑战 |

**应该鼓励的方向**：
- 新偏差校正方法
- 新空间建模方法
- 有物理可解释性的方法

详见 [PM2.5_CMAQ融合方法创新排除.md](./PM2.5_CMAQ融合方法创新排除.md)

---

## 文档索引

| 文档 | 说明 |
|------|------|
| [全流程文档](./PM2.5_CMAQ融合方法自动研究全流程文档_v11_agent_spawn.md) | 核心工作流定义 |
| [十折交叉验证架构文档](./十折交叉验证架构文档.md) | 验证标准 |
| [创新排除规则](./PM2.5_CMAQ融合方法创新排除.md) | 创新判定规则 |
| [评估报告](../DataFusion/评估报告.md) | Downscaler代码评估 |

---

## 常见问题

### Q: 如何添加新的基准方法？

1. 在 `CodeWorkSpace/基准方法代码/` 添加方法实现
2. 运行多阶段验证：`python test_result/基准方法/validate_baseline_multistage.py`
3. 更新 `test_result/基准方法/benchmark_summary.csv`

### Q: 如何跳过某阶段？

在 `test_result/.state/` 创建标记文件：
- `skip_phase1.flag` - 跳过文献下载
- `skip_phase2.flag` - 跳过文献分析

### Q: 如何强制使用特定方案？

创建 `test_result/最终方案标记.json`：
```json
{
  "final": true,
  "method": "RK-Poly",
  "round": 15,
  "note": "人工确认为最优方案"
}
```

---

## 贡献指南

1. Fork本仓库
2. 创建特性分支 `git checkout -b feature/xxx`
3. 提交更改 `git commit -m 'feat: add xxx'`
4. 推送到分支 `git push`
5. 创建Pull Request

---

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@misc{AutoResearch_PM25FusionMethod,
  title = {PM2.5 CMAQ融合方法自动研究系统},
  author = {陈立卓},
  year = {2026},
  url = {https://github.com/zhuobichen/AutoResearch_PM25FusionMethod}
}
```

---

## 许可证

MIT License

---

*最后更新: 2026-04-15*
