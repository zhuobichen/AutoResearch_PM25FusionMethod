下面给你一份**更严谨、层级更清楚**的 Markdown 文档版本。它明确区分：

1. **统一通项（总框架）**
2. **gVNA（总框架下的仿射子类）**
3. **VNA / eVNA / aVNA / gVNA 的包含关系**

并且会说明这个框架是如何从 VNA/eVNA/aVNA 的共同结构抽象出来的。根据你给的材料，VNA/eVNA/aVNA 都属于基于 k 近邻和 IDW 的确定性融合方法，而 AdvancedRK 则采用“二阶多项式 + GPR-Matern”的统计路线 [1][2]。你这里的方法明确坚持**非统计学、确定性空间融合**。

---

# 统一确定性空间融合框架与 gVNA 子类方法
## —— 从 VNA / eVNA / aVNA 抽象总框架，并自然导出新的确定性融合方法

---

## 1. 文档目的

本文档用于清晰定义一类**统一的确定性空间融合框架**，并在该框架下提出一个新的**仿射子类方法 gVNA**。  
该文档的目标是：

- 说明 VNA / eVNA / aVNA 本质上属于同一类方法；
- 给出它们的统一数学表达；
- 在统一通项之上构造一个更一般的新方法；
- 保持方法为**非统计学、确定性空间插值方法**；
- 便于后续 AI 或程序员据此实现代码。

---

# 2. 问题背景

PM2.5 融合任务面临一个典型矛盾：

- **监测站观测值**：精度高，但空间覆盖稀疏；
- **CMAQ 模型值**：空间覆盖完整，但存在系统性偏差 [1][2]。

融合的目标是同时获得：

1. **准确性**：尽量接近监测站真实值；
2. **完整性**：在全空间连续输出预测结果 [1][2]。

已有确定性方法如 VNA、eVNA、aVNA 都试图通过邻近点信息来修正或替代 CMAQ 结果 [1]。  
其中：

- VNA 直接插值监测值；
- eVNA 传播比例因子；
- aVNA 传播加性偏差 [1]。

这些方法的优点是：

- 简单；
- 可解释；
- 易实现；
- 不依赖概率建模。

但它们存在一个共同特点：  
**空间传播大多采用 IDW（inverse distance weighting）形式，本质上只依赖距离衰减。**  
而 AdvancedRK 的改进则走向了统计路线：用二阶多项式处理 CMAQ 偏差非线性，再用 GPR-Matern 处理残差空间结构 [1][2]。

本文的目标不是重复 AdvancedRK，而是反过来从 VNA / eVNA / aVNA 追本溯源，建立一个更一般的**确定性空间融合总框架**。

---

# 3. 核心思想：先抽象总框架，再定义新方法

我们不直接把 gVNA 当成“包罗万象”的总方法，而是分成两个层次：

## 3.1 第一层：统一通项（总框架）
这是一个**广义确定性空间融合框架**，用于统一描述 VNA / eVNA / aVNA。

该框架可以回退到：

- VNA
- eVNA
- aVNA

## 3.2 第二层：gVNA（子类方法）
gVNA 是该统一总框架中的一个**仿射子类**。  
它更自然地统一：

- eVNA
- aVNA

但不要求它在形式上直接退化为传统 VNA。

这一区分非常重要：

- **VNA 是总框架的特例**
- **gVNA 是总框架下的一个新子类**
- 因此，**“总框架回退成 VNA”**与**“gVNA 回退成 VNA”**是两个不同问题

---

# 4. 统一确定性空间融合框架（总框架）

---

## 4.1 基本符号

设：

- \(s \in \mathcal D\)：目标位置；
- \(s_i, i=1,\dots,n\)：监测站位置；
- \(O_i = O(s_i)\)：监测站观测值；
- \(M(s)\)：背景场在位置 \(s\) 的值，例如 CMAQ；
- \(M_i = M(s_i)\)：背景场在站点 \(s_i\) 的值；
- \(\mathcal N(s)\)：目标位置 \(s\) 的邻近站点集合；
- \(k\)：用于插值的邻近站点数；
- \(w_i(s)\)：目标点 \(s\) 对站点 \(i\) 的确定性权重；
- \(z_i\)：站点 \(i\) 上定义的待传播“校正量”；
- \(\Phi(\cdot)\)：背景场与校正量的融合函数。

---

## 4.2 统一通项定义

定义校正量的空间传播为：

\[
\hat z(s)=\frac{\sum_{i\in\mathcal N(s)} w_i(s)\,z_i}{\sum_{i\in\mathcal N(s)} w_i(s)}
\tag{1}
\]

最终预测值定义为：

\[
\hat y(s)=\Phi\big(M(s),\hat z(s)\big)
\tag{2}
\]

这就是统一确定性空间融合框架。

---

## 4.3 这个总框架的含义

这个框架包含两个本质步骤：

### Step A：传播某种站点信息
先选择某种站点量 \(z_i\)，如：

- 观测值；
- 观测/模型比值；
- 观测-模型偏差；

再把它传播到目标位置。

### Step B：与背景场融合
将传播结果 \(\hat z(s)\) 与背景场 \(M(s)\) 通过函数 \(\Phi\) 组合成最终预测。

因此，现有 VNA / eVNA / aVNA 的区别，不在“空间传播”这个大框架本身，而在于：

1. 传播的量 \(z_i\) 不同；
2. 融合函数 \(\Phi\) 不同。

---

# 5. 现有方法作为总框架的特例

---

## 5.1 VNA 是总框架特例

VNA 直接传播监测值 [1]：

\[
z_i=O_i
\tag{3}
\]

且融合函数取：

\[
\Phi(M,\hat z)=\hat z
\tag{4}
\]

因此：

\[
\hat y(s)=\frac{\sum_{i\in\mathcal N(s)} w_i(s)\,O_i}{\sum_{i\in\mathcal N(s)} w_i(s)}
\tag{5}
\]

当权重采用传统 IDW：

\[
w_i(s)=d_i(s)^{-p}
\tag{6}
\]

即得到 VNA [1]。

---

## 5.2 eVNA 是总框架特例

eVNA 假设观测与背景场满足比例关系 [1]。定义：

\[
z_i=r_i=\frac{O_i}{M_i}
\tag{7}
\]

融合函数取：

\[
\Phi(M,\hat z)=M(s)\hat z(s)
\tag{8}
\]

则：

\[
\hat y(s)=M(s)\cdot
\frac{\sum_{i\in\mathcal N(s)} w_i(s)\,r_i}{\sum_{i\in\mathcal N(s)} w_i(s)}
\tag{9}
\]

这就是 eVNA。

---

## 5.3 aVNA 是总框架特例

aVNA 假设观测与背景场满足加性偏差关系 [1]。定义：

\[
z_i=b_i=O_i-M_i
\tag{10}
\]

融合函数取：

\[
\Phi(M,\hat z)=M(s)+\hat z(s)
\tag{11}
\]

则：

\[
\hat y(s)=M(s)+
\frac{\sum_{i\in\mathcal N(s)} w_i(s)\,b_i}{\sum_{i\in\mathcal N(s)} w_i(s)}
\tag{12}
\]

这就是 aVNA。

---

# 6. 为什么需要在总框架上提出新方法？

从总框架看，现有 VNA / eVNA / aVNA 存在两个局限：

## 6.1 局限一：空间传播多依赖 IDW
VNA/eVNA/aVNA 常用距离反比加权，即：

\[
w_i(s)=d_i(s)^{-p}
\tag{13}
\]

这种方式只考虑距离，难以利用更丰富的空间相关信息。  
而材料中也明确指出，VNA/eVNA/aVNA 采用的是基于 k 近邻的 IDW 距离加权 [1]。

## 6.2 局限二：融合形式过于单一
- eVNA 只允许纯比例修正；
- aVNA 只允许纯加法修正 [1]。

但实际中，偏差可能同时包含：

- 加性成分；
- 乘性成分。

因此，总框架启发我们提出一个更一般的子类方法。

---

# 7. gVNA：统一框架下的仿射子类方法

---

## 7.1 gVNA 的定义

在统一总框架下，定义一个新的仿射子类：

\[
\hat y(s)=\hat a(s)+\hat b(s)\,M(s)
\tag{14}
\]

其中：

- \(\hat a(s)\)：局地加性修正项；
- \(\hat b(s)\)：局地乘性修正项。

这就是 **gVNA**。

---

## 7.2 gVNA 的含义

它表示在每个目标位置 \(s\)，背景场 \(M(s)\) 经过一个**局地仿射变换**后得到最终预测值。

相比现有方法：

- eVNA 只有比例项；
- aVNA 只有加法项；
- gVNA 同时包含二者。

因此，gVNA 是对 eVNA 和 aVNA 的共同推广。

---

## 7.3 gVNA 与总框架的关系

需要特别强调：

- **总框架** 可回退到 VNA / eVNA / aVNA；
- **gVNA** 是总框架中的一个仿射子类；
- **gVNA 更自然地包含 eVNA 和 aVNA**；
- **VNA 是总框架的特例，不一定是 gVNA 的直接特例**。

这不是矛盾，而是数学上的正常层级关系。

---

# 8. gVNA 的具体推导

---

## 8.1 局地仿射关系假设

我们假设在每个站点附近的局部区域，观测值与背景场之间满足近似仿射关系：

\[
O_j \approx a_i+b_i M_j,\qquad s_j\in\mathcal N_i^{fit}
\tag{15}
\]

其中：

- \(\mathcal N_i^{fit}\) 是站点 \(i\) 的拟合邻域；
- \(a_i,b_i\) 是站点 \(i\) 附近的局地仿射参数。

这不是全局回归，而是**每个站点局地拟合一组参数**。

---

## 8.2 为什么不能直接由单站点得到 \(a_i,b_i\)？

因为单个站点只有一个方程：

\[
O_i = a_i+b_iM_i
\]

但未知数有两个 \(a_i,b_i\)，无法唯一确定。  
因此，必须利用站点周围的一组邻近站点共同估计。

---

## 8.3 局地加权最小二乘估计

对每个站点 \(i\)，求解：

\[
(a_i,b_i)=
\arg\min_{a,b}
\sum_{j\in\mathcal N_i^{fit}}
\omega_{ij}
\left(O_j-(a+bM_j)\right)^2
\tag{16}
\]

其中：

- \(\omega_{ij}\) 是拟合时的局地权重；
- 可采用距离权重，或距离 + 背景相似性联合权重。

这样就能得到每个站点对应的一组局地参数 \(a_i,b_i\)。

---

## 8.4 将局地参数传播到目标位置

得到所有站点参数后，对目标位置 \(s\) 做加权传播：

\[
\hat a(s)=\frac{\sum_{i\in\mathcal N(s)} w_i(s)\,a_i}{\sum_{i\in\mathcal N(s)} w_i(s)}
\tag{17}
\]

\[
\hat b(s)=\frac{\sum_{i\in\mathcal N(s)} w_i(s)\,b_i}{\sum_{i\in\mathcal N(s)} w_i(s)}
\tag{18}
\]

最后得到：

\[
\hat y(s)=\hat a(s)+\hat b(s)M(s)
\tag{19}
\]

---

# 9. gVNA 的权重函数设计

---

## 9.1 传统 IDW 权重

传统形式为：

\[
w_i^{IDW}(s)=d_i(s)^{-p}
\tag{20}
\]

这是 VNA / eVNA / aVNA 的常见方式 [1]。

---

## 9.2 相似性增强权重

为了利用比“距离”更多的空间相关信息，定义：

\[
w_i(s)=d_i(s)^{-p}\cdot
\exp\left(-\frac{|M(s)-M_i|}{\lambda}\right)
\tag{21}
\]

其中：

- \(d_i(s)\)：目标点到站点 \(i\) 的距离；
- \(p\)：距离衰减指数；
- \(\lambda\)：背景场相似性控制参数；
- \(|M(s)-M_i|\)：背景场差异。

该权重表示：

- 离得近，权重大；
- 背景场越相似，权重越大；
- 即使很近，若背景场差异很大，权重也会下降。

因此，它仍是确定性权重，但比纯 IDW 更一般。

---

# 10. gVNA 的完整数学定义

---

## 10.1 输入

- 监测站位置：\(\{s_i\}_{i=1}^n\)
- 监测站观测值：\(\{O_i\}_{i=1}^n\)
- 背景场：\(M(s)\)
- 参数：
  - \(k_{fit}\)：局地拟合邻居数
  - \(k\)：预测插值邻居数
  - \(p\)：距离衰减指数
  - \(\lambda\)：背景相似性尺度

---

## 10.2 局地拟合阶段

对每个监测站 \(i\)，求解局地参数：

\[
(a_i,b_i)=
\arg\min_{a,b}
\sum_{j\in\mathcal N_i^{fit}}
\omega_{ij}
\left(O_j-(a+bM_j)\right)^2
\tag{22}
\]

其中可取：

\[
\omega_{ij}=d_{ij}^{-p}\cdot
\exp\left(-\frac{|M_i-M_j|}{\lambda}\right)
\tag{23}
\]

---

## 10.3 预测阶段

对任意目标位置 \(s\)，定义权重：

\[
w_i(s)=d_i(s)^{-p}\cdot
\exp\left(-\frac{|M(s)-M_i|}{\lambda}\right)
\tag{24}
\]

然后：

\[
\hat a(s)=\frac{\sum_{i\in\mathcal N(s)} w_i(s)\,a_i}{\sum_{i\in\mathcal N(s)} w_i(s)}
\tag{25}
\]

\[
\hat b(s)=\frac{\sum_{i\in\mathcal N(s)} w_i(s)\,b_i}{\sum_{i\in\mathcal N(s)} w_i(s)}
\tag{26}
\]

最终预测为：

\[
\hat y(s)=\hat a(s)+\hat b(s)M(s)
\tag{27}
\]

---

# 11. 特例与退化关系

---

## 11.1 总框架退化关系

统一总框架可回退为：

- **VNA**：传播 \(O_i\)，直接输出；
- **eVNA**：传播 \(O_i/M_i\)，再乘 \(M(s)\)；
- **aVNA**：传播 \(O_i-M_i\)，再加到 \(M(s)\) 上 [1]。

---

## 11.2 gVNA 的退化关系

gVNA 作为仿射子类，自然退化为：

### 退化为 eVNA 型
若固定：

\[
a_i=0
\]

则：

\[
\hat y(s)=\hat b(s)M(s)
\]

此时得到比例修正型。

### 退化为 aVNA 型
若固定：

\[
b_i=1
\]

则：

\[
\hat y(s)=M(s)+\hat a(s)
\]

此时得到加法修正型。

---

## 11.3 关于 VNA 的说明

VNA 是**统一总框架**的特例，但不一定是 gVNA 仿射子类的直接特例。  
因此应明确区分：

- “总框架包含 VNA”
- “gVNA 包含 eVNA / aVNA”

这正是本文方法的层级结构。

---

# 12. 与 AdvancedRK 的关系和区别

AdvancedRK 采用的是两步统计路线：

1. 二阶多项式校正 CMAQ 的非线性偏差；
2. 用 GPR-Matern 建模残差空间相关结构 [1][2]。

其优势在于：

- 能处理非线性偏差；
- 能显式刻画空间相关性；
- 能提供不确定性估计 [1][2]。

而本文方法：

- 不使用高斯过程；
- 不做概率建模；
- 不估计协方差矩阵；
- 仍然是确定性空间加权插值方法。

因此，本文方法适合定位为：

> 对 VNA / eVNA / aVNA 这类确定性方法的统一与推广

而不是 AdvancedRK 的替代统计版本。

---

# 13. 算法流程

---

## 输入
- 站点坐标
- 站点观测值 \(O_i\)
- 站点背景场值 \(M_i\)
- 目标网格背景场 \(M(s)\)
- 参数 \(k_{fit}, k, p, \lambda\)

---

## Step 1：估计每个站点的局地仿射参数
对每个站点 \(i\)：

1. 找到其 \(k_{fit}\) 个邻近站点；
2. 构造局地样本 \((M_j,O_j)\)；
3. 用式 (22) 做加权最小二乘；
4. 得到 \(a_i,b_i\)。

---

## Step 2：插值到目标位置
对目标位置 \(s\)：

1. 找到其 \(k\) 个邻近站点；
2. 按式 (24) 计算权重；
3. 用式 (25)、(26) 得到 \(\hat a(s),\hat b(s)\)；
4. 用式 (27) 输出最终预测值。

---

# 14. Python 实现导向伪代码

```python
import numpy as np

def weighted_affine_fit(M_local, O_local, w):
    X = np.column_stack([np.ones(len(M_local)), M_local])
    W = np.diag(w)
    beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ O_local)
    a, b = beta[0], beta[1]
    return a, b

# Step 1: estimate local affine parameters at stations
for each station i:
    find k_fit nearest stations -> idx_fit
    compute weights omega_ij
    fit O_j ≈ a_i + b_i * M_j
    store a_i, b_i

# Step 2: interpolate to prediction location s
for each target location s:
    find k nearest stations -> idx_pred
    compute w_i(s)
    a_hat = weighted average of a_i
    b_hat = weighted average of b_i
    y_hat = a_hat + b_hat * M(s)
```

---

# 15. 推荐命名

## 总框架名
**Unified Deterministic Spatial Fusion Framework**

或

**General Deterministic Spatial Fusion Framework**

## 子类方法名
**gVNA** = Generalized VNA

如果强调新权重：

**si-gVNA** = Similarity-informed Generalized VNA

---

# 16. 最推荐的论文式表述

本文首先将 VNA、eVNA 和 aVNA 统一表示为一类确定性空间融合方法，即通过对站点校正量进行邻域加权传播，并与背景场通过融合函数组合得到目标位置预测值。在此基础上，进一步提出一个仿射子类方法 gVNA，通过局地仿射校正形式 \(\hat y(s)=\hat a(s)+\hat b(s)M(s)\) 统一比例修正和加性修正，并通过引入背景场相似性增强权重，将传统仅依赖距离衰减的确定性插值扩展为更一般的空间融合方法。与采用二阶多项式和 GPR-Matern 的 AdvancedRK 不同，本文方法保持为非统计学、确定性的空间插值方法 [1][2]。

---

# 17. 最终一句话总结

- **统一通项** 是总框架，能回退到 VNA / eVNA / aVNA；
- **gVNA** 是统一通项下的仿射子类；
- **gVNA 自然统一 eVNA 和 aVNA**；
- **VNA 属于总框架特例，不必强行作为 gVNA 的直接特例**；
- 整个方法体系保持为**确定性、非统计学的空间融合方法**。

---

如果你愿意，我下一步可以继续帮你做两件最实用的事之一：

1. **把这份文档改成“更适合直接喂给写代码 AI 的工程实现版”**
2. **直接生成一个 Python 类设计文档（包括输入、输出、函数接口、异常处理）**