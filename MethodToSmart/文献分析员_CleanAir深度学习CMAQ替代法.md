# 【可执行方法规范】

## 方法名称
CleanAir - 深度学习CMAQ替代模型 (Deep Learning CTM Emulator)

## 文献来源
- 论文标题: "A deep-learning model for predicting daily PM2.5 concentration in response to emission reduction"
- 作者: Shigan Liu, Guannan Geng, Yanfei Xiang, Hejun Hu, Xiaodong Liu, Xiaomeng Huang, Qiang Zhang
- 机构: 清华大学
- 日期: 2025
- arXiv: 2506.18018

## 核心公式

### 模型输入 (4类):
1. **排放清单**: 8类污染物（PM2.5, BC, OC, PM10, SO2, NOx, NH3, NMVOCs）
2. **气象场**: 温度、湿度、风速、风向、降水、边界层高度
3. **时间编码**: 年、月、日、季节
4. **空间编码**: 经纬度

### 网络架构: Residual Symmetric 3D U-Net

**编码器部分:**
- Conv3D + GroupNorm + ELU
- MaxPooling下采样
- 5个残差模块

**解码器部分:**
- Up-sampling上采样
- Skip connections (求和连接)
- 输出: PM2.5浓度场

### 损失函数 (多任务学习):
$$
L = \sum_k w_k \cdot L_k
$$
其中 $w_k$ 是自适应权重，自动平衡不同任务的贡献。

### 残差连接:
$$
y = F(x) + x
$$
保持梯度流，改善训练稳定性。

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| learning_rate | float | 1e-4 | 学习率 |
| batch_size | int | 16 | 批大小 |
| epochs | int | 100 | 训练轮数 |
| optimizer | str | 'Adam' | 优化器 |
| weight_decay | float | 1e-5 | 权重衰减 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 说明 |
|-----|------|-----|------|
| 排放数据 | array | (n_time, n_lat, n_lon, n_species) | 8类污染物 |
| 气象场 | array | (n_time, n_lat, n_lon, n_var) | 6个变量 |
| 时间编码 | array | (n_time, 4) | 年月日季节 |
| 空间坐标 | array | (n_lat, n_lon, 2) | 经纬度 |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| PM2.5浓度场 | array | μg/m³ |

## 实现步骤

1. **CMAQ模拟**: 生成基线和多种排放削减情景的训练数据
2. **数据预处理**: 标准化输入变量
3. **网络构建**: 构建Residual 3D U-Net
4. **多任务损失**: 设计自适应权重损失函数
5. **训练**: 使用Adam优化器训练
6. **推理**: 9秒完成全年模拟（单GPU）

## 性能指标
- 速度: 比CMAQ快5个数量级（9秒 vs 数天）
- 精度: R = 0.73-0.94，RMSE = 8.07-21.25 μg/m³
- 泛化: 可预测未见过的气象条件和排放情景

## 随机性
- [ ] 是（深度学习训练带随机初始化）

## 方法指纹
MD5: cleanair_dl_ctm_emulator

## 实现检查清单
- [ ] 核心公式已验证（论文方法部分）
- [ ] 网络架构已实现
- [ ] 损失函数已实现
