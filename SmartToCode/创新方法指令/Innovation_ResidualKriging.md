# 创新方法指令

## 方法名称
Residual Kriging with Adaptive Variogram - 自适应变异函数残差克里金

## 创新核心
使用自适应变异函数进行残差克里金插值，比简单IDW更精确地建模空间相关性

## 核心公式
$$
P_{RK}(s_0) = M(s_0) + R^*(s_0)
$$
其中 $R^*$ 是使用变异函数γ(h)建模的克里金残差

### 变异函数模型
$$
\gamma(h) = c_0 + c \cdot \left(1 - e^{-\frac{h}{a}}\right)
$$
- c0: 块金效应（nugget）
- c: 拱高（sill）
- a: 变程（range）

### 克里金权重
$$
w = \Gamma^{-1} \cdot \gamma
$$
其中 $\Gamma_{ij} = \gamma(|s_i - s_j|)$

## 关键步骤
1. 计算训练站点的残差：$R(s_i) = O(s_i) - M(s_i)$
2. 使用残差拟合变异函数参数
3. 构建克里金矩阵并求解
4. 对网格点进行克里金预测
5. 融合：$P = M + R^*$

## 创新优势
- 相比简单IDW，克里金考虑了空间相关性
- 自适应变异函数可以更好捕捉不同尺度的空间结构
- 预期R²提升 ≥ 0.01

## 方法指纹
MD5: residual_kriging_adaptive_variogram_v1

## 实现注意事项
- 使用scipy的变异函数模型
- 对于大矩阵使用近似方法加速