# 方法名称
克里金信息扩散降尺度法 (Kriging-informed Conditional Diffusion Probabilistic Model, Ki-CDPM)

## 类型
复现

## 核心公式

### 1. 条件扩散模型框架
扩散模型通过去噪过程学习条件分布 $p(y|x)$，其中 $x$ 为低分辨率输入，$y$ 为高分辨率目标。

前向过程（加噪）：
$$
q(y_t | y_{t-1}) = \mathcal{N}(y_t; \sqrt{1-\beta_t} y_{t-1}, \beta_t I)
$$

反向过程（去噪）：
$$
p_\theta(y_{t-1} | y_t, x) = \mathcal{N}(\mu_\theta(y_t, x, t), \sigma_t^2 I)
$$

### 2. Kriging约束项
将空间克里金插值的协方差结构融入扩散模型的注意力机制：

空间协方差矩阵（基于变异函数）：
$$
\Sigma_{ij} = \gamma(h_{ij}) = \begin{cases} 0 & h = 0 \\ C(0) - C(h) & h > 0 \end{cases}
$$

其中 $h_{ij}$ 是两点间的距离，$\gamma(h)$ 是变异函数。

### 3. Kriging条件均值
对于低分辨率网格点 $x_i$，Kriging预测：
$$
\hat{y}_{Krig}(x_i) = \sum_{j=1}^{n} \lambda_j C(x_i, x_j)
$$
其中 $\lambda_j$ 是Kriging权重，$C$ 是协方差函数。

### 4. 条件引导机制
将Kriging信息编码为扩散模型的条件输入：
$$
\epsilon_\theta(y_t, x, t, K_{spatial}) = \epsilon_\theta(y_t, x, t) + w \cdot \nabla_{y_t} \log p(K_{spatial}|y_t)
$$

## 算法步骤

1. **阶段1：空间协方差结构学习**
   - 使用训练数据估计变异函数参数 $\gamma(h)$
   - 构建空间协方差矩阵 $\Sigma$
   - 计算Kriging权重矩阵 $K$

2. **阶段2：条件扩散模型构建**
   - 构建U-Net骨干网络
   - 编码Kriging空间结构为额外条件通道
   - 设计时空注意力机制融入Kriging核

3. **阶段3：训练与推理**
   - 条件扩散训练：$p_\theta(y|x, K)$
   - 多步去噪生成高分辨率场
   - 可选：多次采样产生集合预报

## 参数说明
- diffusion_steps：扩散模型时间步数，默认1000
- beta_schedule：噪声调度方案，默认linear
- spatial_kernel：空间核函数类型，默认gaussian
- range_param：变异函数变程参数，由数据驱动估计
- nugget：变异函数块金效应，默认0
- guidance_weight w：Kriging引导权重，默认0.1-0.5
- backbone：扩散模型骨干网络，默认U-Net

## 预期效果
- R2：0.78-0.85（相对于传统统计降尺度）
- RMSE：15-20 μg/m³
- MB：±3 μg/m³

## 验证方案
十折交叉验证：将研究区域划分为10个空间区块，轮流使用9块训练、1块验证，评估空间外推能力。
