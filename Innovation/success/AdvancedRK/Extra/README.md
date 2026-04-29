# AdvancedRK 补充实验

## 文件说明

| 文件 | 说明 |
|------|------|
| `01_kernel_comparison_sensitivity.py` | 核函数敏感性实验：比较RBF/Matern/RationalQuad/RF/KNN |
| `02_polynomial_degree_sensitivity.py` | 多项式阶数敏感性实验：比较1/2/3阶多项式 |
| `04_step2_only_experiment.py` | 跳过多项式实验：Step 2 Only vs Step 1+2 |
| `03_ablation_experiment_report.md` | 消融实验报告：汇总四组实验结果 |
| `learn_advancedrk_from_scratch.html` | **学习指南（从零开始v2）**：新增VNA/eVNA/aVNA/Downscaler基准方法详解 |
| `learn_advancedrk.html` | **学习指南（进阶版）**：面向有一定基础的读者，侧重应用 |

## 实验列表

### 实验1：核函数敏感性（01_kernel_comparison_sensitivity.py）
比较不同核函数和其他ML方法的效果

### 实验2：多项式阶数敏感性（02_polynomial_degree_sensitivity.py）
比较不同阶数多项式与GPR结合的效果

### 实验3：Step消融（03_ablation_experiment_report.md）
Step 1（多项式）vs Step 1+2（多项式+GPR）的对比

### 实验4：跳过多项式（04_step2_only_experiment.py）
直接用GPR建模（不经过多项式）vs 完整流程