# 创新方法指令

## 方法名称
Hybrid eVNA-aVNA (混合比率-偏差法)

## 创新核心
结合eVNA和aVNA的优点，根据站点偏差大小动态选择校正方式

## 核心公式
$$
P_{Hybrid}(s_0) = M(s_0) \times \alpha(s_0) + (1-\alpha(s_0)) \times M(s_0)
$$
其中 $\alpha$ 是动态权重，基于邻近站点的偏差一致性计算

简化版：
$$
P_{Hybrid}(s_0) = \beta \cdot P_{eVNA}(s_0) + (1-\beta) \cdot P_{aVNA}(s_0)
$$
其中 $\beta$ 通过交叉验证优化

## 关键步骤
1. 计算训练集站点的偏差和比率
2. 用IDW插值偏差和比率到网格
3. 分别计算eVNA和aVNA结果
4. 混合系数β通过十折验证优化
5. 最终融合结果 = β × eVNA + (1-β) × aVNA

## 创新优势
- 结合eVNA（比率法）和aVNA（偏差法）的优点
- β通过数据驱动确定
- 预期R²提升 ≥ 0.01

## 方法指纹
MD5: hybrid_evna_avna_fingerprint

## 输入输出格式
- 输入：监测数据CSV + CMAQ netCDF
- 输出：融合网格PM2.5
- 支持十折交叉验证
