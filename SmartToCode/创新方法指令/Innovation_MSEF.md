# 创新方法指令

## 方法名称
Multi-Scale Ensemble Fusion (MSEF) - 多尺度集成融合

## 创新核心
结合eVNA、GMOS、Downscaler三种方法的优点，通过加权集成提升预测精度

## 核心公式
$$
P_{MSEF}(s_0) = \beta_1 \cdot P_{eVNA}(s_0) + \beta_2 \cdot P_{GMOS}(s_0) + \beta_3 \cdot P_{Downscaler}(s_0)
$$
其中 $\beta_1 + \beta_2 + \beta_3 = 1$，权重通过十折交叉验证优化

## 各方法说明

### eVNA (Enhanced VNA)
$$
P_{eVNA}(s_0) = M(s_0) \times \sum_i w_i \cdot \frac{O(s_i)}{M(s_i)}
$$
- 基于比率法的空间插值
- k=30近邻，power=-2

### GMOS (Gridded Model Output Statistics)
多尺度半径迭代校正：
```
for r in radii:
    w_i = s_i * (r² - d_i²) / (r² + d_i²)
    C = sum(w_i * (O_i - A_i))
    A += C
```
- 使用多尺度半径（2000km, 1000km, 500km, 250km, 125km, 62km, 31km, 15km）
- 迭代校正模型偏差

### Downscaler (Kriging残差插值)
$$
P_{Downscaler}(s_0) = M(s_0) + Kriging(O(s_i) - M(s_i))
$$
- 使用克里金插值校正残差

## 关键步骤
1. 在训练站点计算三种方法的预测值
2. 使用十折验证优化权重β
3. 最终融合结果使用最优权重组合

## 创新优势
- 结合三种不同原理的方法（比率法、多尺度迭代、克里金）
- 集成学习理论表明，多样性模型集成可提升泛化能力
- 预期R²提升 ≥ 0.01

## 方法指纹
MD5: msef_method_fingerprint_v1

## 输入输出格式
- 输入：监测数据CSV + CMAQ netCDF
- 输出：融合网格PM2.5
- 支持十折交叉验证

## 实现注意事项
1. GMOS方法在Code/VNAeVNAaVNA/nna_methods/__init__.py中已实现
2. 可复用NNA类进行eVNA计算
3. 权重优化使用网格搜索（β1, β2, β3 ∈ [0,1]，步长0.1）