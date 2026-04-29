# PM2.5 CMAQ融合方法自动研究系统 - 项目清单

> 生成时间: 2026-04-21
> 整理版本: v2.0

---

## 一、根目录结构（整理后）

```
E:\CodeProject\ClaudeRoom\Data_Fusion_AutoResearch\
├── CLAUDE.md                    # Claude Code项目说明
├── INVENTORY.md                 # 项目总清单（本文件）
├── run_pipeline.py              # 工作流启动脚本
├── .claude/                     # Claude Code配置
├── .git/                        # Git版本控制
├── .gitignore
├── .agent_state.json
│
├── PaperDownload/               # 论文PDF文件
├── PaperDownloadMd/             # 论文清单、分析报告
├── LocalPaperLibrary/           # 本地原始论文库
├── MethodToSmart/               # 文献分析员输出（方法分析文档）
├── SmartToCode/                 # 方案设计师输出（实现指令）
├── Code/                        # 参考代码（Downscaler/VNAeVNAaVNA）
├── CodeWorkSpace/               # 工作区代码
│   ├── 基准方法代码/            # VNA/eVNA/aVNA变体
│   ├── 复现方法代码/            # 复现方法实现
│   ├── 新融合方法代码/          # 创新方法实现
│   ├── 年均融合方法/            # 年均数据融合方法
│   └── 改造后VNA_eVNA_aVNA/     # 改造后VNA系列方法
├── test_data/                   # 测试数据
├── test_result/                 # 测试结果
│   ├── 基准方法/                # 基准方法验证结果
│   ├── 创新方法/                # 创新方法验证结果
│   ├── 历史/                    # 历史验证结果
│   ├── snapshots/              # 状态快照
│   ├── InnovationMethods/       # 创新方法代码
│   ├── legacy_tests/           # 历史测试脚本（已归档）
│   └── 代码实现报告.md          # 代码实现清单
├── Innovation/                  # 已确认创新方法
│   ├── success/                 # 验证通过的方法
│   │   ├── AdvancedRK/          # AdvancedRK（R²=0.9162，最优）
│   │   ├── PolyRK/              # PolyRK（R²=0.9105，核心创新）
│   │   └── RobustRK/            # RobustRK
│   └── failed/                  # 验证失败的方法
│       ├── ARK_OLS/
│       ├── CGARK/
│       ├── GARK/
│       ├── MSAGARK/
│       ├── PG-STGAT/
│       └── VCFFM/
├── paper_output/                # 论文输出
│   ├── paper.tex                # 论文主文件
│   ├── paper.pdf                # 编译后PDF
│   ├── references.bib           # 参考文献
│   ├── figures/                 # 论文图表
│   └── README.md                # 项目说明（由根目录迁移）
├── agents/                       # Agent模块
│   ├── spawn_executor.py        # Agent spawn执行器
│   ├── role_templates.py       # 角色prompt模板
│   ├── workflow_orchestrator.py
│   ├── research_state_tracker.py
│   └── ...
├── error/                        # 错误日志
│   ├── diagnosis_*.md           # 诊断报告
│   ├── *.log                    # 运行日志
│   └── temp_*.txt               # 临时文件（已归档）
├── skills/                       # Claude Code Skills
├── 文档拆分/                     # 项目文档拆分
│   ├── 01_项目概览与核心原则.md
│   ├── 02_创新判定规范.md
│   ├── 03_Agent_Spawn工作流.md
│   ├── 04_输出与快照规范.md
│   ├── 05_运行与异常处理.md
│   ├── 06_系统目录结构.md
│   ├── PM2.5_CMAQ融合方法自动研究全流程文档_v11_agent_spawn.md
│   └── 十折交叉验证架构文档.md
└── LizhuoChen/                   # 用户个人代码（保留）
```

---

## 二、本次整理记录

### 2.1 移至正确位置的文件

| 原位置 | 新位置 | 原因 |
|--------|--------|------|
| 根目录/*.log | error/ | 禁止根目录放日志文件 |
| 根目录/temp_*.txt | error/ | 禁止根目录放临时文件 |
| 根目录/README.md | paper_output/ | 禁止根目录放独立文档 |
| 根目录/十折交叉验证架构文档.md | 文档拆分/ | 禁止根目录放架构文档 |
| 根目录/PM2.5_CMAQ融合方法..._v11_agent_spawn.md | 文档拆分/ | 禁止根目录放独立文档 |
| CodeWorkSpace/*_验证.py (14个) | test_result/legacy_tests/ | 禁止根目录放测试脚本 |
| CodeWorkSpace/代码实现报告.md | test_result/ | 禁止根目录放报告文件 |

### 2.2 删除的目录

| 目录 | 原因 |
|------|------|
| CodeWorkSpace/WorkDocument/ | 空目录，已删除 |

---

## 三、目录完整性检查

| 必需目录 | 状态 | 说明 |
|----------|------|------|
| PaperDownload/ | ✅ 存在 | 论文PDF |
| PaperDownloadMd/ | ✅ 存在 | 论文清单 |
| LocalPaperLibrary/ | ✅ 存在 | 本地论文库 |
| MethodToSmart/ | ✅ 存在 | 文献分析 |
| SmartToCode/ | ✅ 存在 | 方案设计 |
| Code/ | ✅ 存在 | 参考代码 |
| CodeWorkSpace/ | ✅ 存在 | 工作区代码 |
| test_data/ | ✅ 存在 | 测试数据 |
| test_result/ | ✅ 存在 | 测试结果 |
| Innovation/ | ✅ 存在 | 已确认创新 |
| paper_output/ | ✅ 存在 | 论文输出 |
| agents/ | ✅ 存在 | Agent模块 |
| error/ | ✅ 存在 | 错误日志 |
| .claude/ | ✅ 存在 | Claude配置 |
| 文档拆分/ | ✅ 存在 | 项目文档 |

**禁止在根目录的文件类型**（本次已清理）：
- 临时文件（temp_*.txt）
- 日志文件（*.log）
- 测试脚本（*_验证.py, *十折*.py, test_*.py）
- 独立文档（*.md, *架构文档.md, *排除.md）
- 报告文件（comparison_report.md, 代码实现报告.md）

---

## 四、核心方法清单

### 4.1 基准方法（Baseline）

| 方法 | 目录 | 说明 |
|------|------|------|
| VNA | Code/VNAeVNAaVNA/ | Voronoi Neighbor Average |
| eVNA | Code/VNAeVNAaVNA/ | 乘法偏差校正 |
| aVNA | Code/VNAeVNAaVNA/ | 加法偏差校正 |
| Downscaler | Code/Downscaler/ | MCMC降尺度 |

### 4.2 已确认创新方法（Innovation/success/）

| 方法 | stage1 R² | stage2 R² | stage3 R² | 状态 |
|------|-----------|-----------|-----------|------|
| **AdvancedRK** | 0.9162 | 0.8526 | 0.9129 | ✅ 4/4通过（最优） |
| **PolyRK** | 0.9105 | 0.8474 | 0.9060 | ✅ 4/4通过（核心） |
| RobustRK | ~0.91 | - | - | 部分验证 |

### 4.3 验证失败方法（Innovation/failed/）

| 方法 | 失败原因 |
|------|----------|
| GARK | IDW类无明确优势 |
| CGARK | IDW类无明确优势 |
| MSAGARK | IDW类无明确优势 |
| PG-STGAT | 图网络路线验证失败 |
| VCFFM | 验证失败 |
| ARK_OLS | 验证失败 |
| BayesianVariationalFusion | 验证失败 |

### 4.4 排除方法（不测试）

| 方法 | 排除原因 |
|------|----------|
| PSK | 样条校正无实质创新 |
| CSPRK | 浓度分层不合理 |
| Stacking类 | 加权集成，迁移性差 |

---

## 五、基准阈值（VNA方法）

| 阶段 | 时间范围 | R² > | RMSE ≤ | \|MB\| ≤ |
|------|----------|-------|--------|----------|
| pre_exp | 2020-01-01~05 | 0.8907 | 16.68 | 0.70 |
| stage1 | 2020-01 | 0.9034 | 16.48 | 0.50 |
| stage2 | 2020-07 | 0.8408 | 5.05 | 0.05 |
| stage3 | 2020-12 | 0.9031 | 12.20 | 0.42 |

---

## 六、快速命令

```bash
# 查看项目状态
python run_pipeline.py --status

# 运行基准方法多阶段验证
python test_result/基准方法/validate_baseline_multistage.py

# 运行创新方法十折验证
python test_result/创新方法/PolyRK_十折标准模式.py
python test_result/创新方法/AdvancedRK_十折标准模式.py

# 运行所有创新方法验证
python test_result/创新方法/validate_all_methods.py
```

---

*本清单由项目整理智能体自动生成*
*整理时间: 2026-04-21*
