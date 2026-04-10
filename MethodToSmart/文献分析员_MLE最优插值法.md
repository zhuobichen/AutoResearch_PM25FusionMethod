# 【可执行方法规范】

## 方法名称
MLE-OI - 最大似然估计最优插值法 (Maximum Likelihood Optimal Interpolation)

## 文献来源
- 论文标题: "Model evaluation and spatial interpolation by Bayesian combination of observations with numerical models" (Fuentes and Raftery, 2005)
- 方法: 贝叶斯组合观测与数值模型输出

## 核心公式

### 简化最优插值:
$$
\hat{y}_{OI}(s_0) = \mathbf{x}(s_0)^T \beta + \sum_{i=1}^{n} w_i (y_i - \mathbf{x}(s_i)^T \beta)
$$

### 权重确定（最小化估计方差）:
$$
\mathbf{w} = (\mathbf{X}^T \mathbf{R}^{-1} \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{R}^{-1}
$$

### 偏差校正形式:
$$
\hat{y}_{final}(s) = (1 - k) \cdot CMAQ(s) + k \cdot \hat{y}_{OI}(s)
$$
其中k是融合权重

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| lambda | float | 1.0 | 正则化参数 |
| correlation_scale | float | data-fitted | 空间相关尺度 |

## 方法指纹
MD5: mle_optimal_interpolation_method

## 实现检查清单
- [ ] 核心公式已验证
- [ ] 权重计算已实现
