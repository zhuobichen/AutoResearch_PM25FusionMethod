# test_result 清单

生成时间：2026-04-10

## 目录结构

```
test_result/
├── INVENTORY.md              # 本清单
├── .state/                   # 状态追踪
├── 基准方法/                  # 2 文件
├── 复现方法/                 # 0 文件
├── 创新方法/                 # 45 个方法_summary.csv
├── 历史最佳方案/             # 当前最优方案
├── 历史/                     # 历史归档
│   └── 年平均融合测试/
├── comparison_report.md      # 对比报告
└── FINAL_REPORT.md         # 最终报告
```

## 基准方法指标

| 方法 | R² | RMSE | MAE | MB |
|------|-----|------|-----|-----|
| CMAQ | -0.0376 | 29.25 | 20.47 | -3.24 |
| VNA | 0.7996 | 12.86 | 7.75 | 0.76 |
| aVNA | 0.7941 | 13.03 | 8.10 | 0.10 |
| eVNA | 0.8100 | 12.52 | 7.99 | 0.08 |

## 复现方法 (0 个)

| 方法名 | 状态 |
|--------|------|

## 创新方法 (45 个)

| 方法名 | R² | 状态 |
|--------|-----|------|
| BayesianVariationalFusion | 10.0000 | 待验证 |
| SpatialZoneEnsemble | 4.0000 | 待验证 |
| MultiKernelGPREnsemble | 0.9200 | 待验证 |
| CSPRK | 0.9135 | 待验证 |
| GradientBoostingEnsemble | 0.9000 | 待验证 |
| NNResidualEnsemble | 0.9000 | 待验证 |
| PolyEnsemble | 0.9000 | 待验证 |
| SuperEnsemble | 0.9000 | 待验证 |
| TripleEnsemble | 0.9000 | 待验证 |
| HGPRK | 0.8519 | 待验证 |
| ... | 共 38 个方法 |

## 状态追踪

| 文件 | 说明 |
|------|------|
| .state/ledger.jsonl | 决策记录 |
| .state/research_status.md | 研究状态 |

## 规范说明

- **一角色一清单**：每个角色只在自己目录生成一份清单
- **一次一版本**：报告类文件只保留最新版本
- **机器可读**：汇总数据必须是 CSV/JSON
- **人类可读**：报告必须是 Markdown/PDF
