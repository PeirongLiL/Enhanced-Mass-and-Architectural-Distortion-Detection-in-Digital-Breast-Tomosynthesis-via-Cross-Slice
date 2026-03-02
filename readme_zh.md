# Enhanced Mass and Architectural Distortion Detection in Digital Breast Tomosynthesis via Cross-Slice Weighted Box Fusion

[**English README**](README.md)

## 项目简介

本仓库提供论文：

**“Enhanced Mass and Architectural Distortion Detection in Digital Breast Tomosynthesis via Cross-Slice Weighted Box Fusion”**  
（投稿至 *Pattern Recognition*，审稿中）

的官方代码实现。

本研究提出了一套面向 **数字乳腺断层合成（Digital Breast Tomosynthesis, DBT）** 的检测与评估框架。通过引入创新的 **跨切片加权框融合算法（Cross-Slice Weighted Box Fusion, DBT_WBF）**，提升肿块（Mass）与结构扭曲（Architectural Distortion）的检测性能。

该框架包括：

- 数据集构建
- 基于 YOLOv8 的模型训练
- 推理流程
- AUC 与 FROC 评估
- 跨切片检测框融合

本方法充分利用 DBT 三维体数据的结构特性，通过融合相邻切片的预测结果，提高检测稳定性并降低假阳性率。

---

## 项目结构

### `config.py`

训练与推理的配置文件。

包括：

- 模型参数
- 训练超参数
- 数据路径配置
- 推理阈值设置

检测模型基于 **YOLOv8**，使用 **MMYOLO / MMDetection** 框架进行训练：  
https://mmyolo.readthedocs.io/zh-cn/latest/

---

## 数据集准备

本研究使用 **BCS-DBT 数据集**：  
https://sites.duke.edu/mazurowski/resources/digital-breast-tomosynthesis-database/

### 数据生成脚本

#### `Generate_training_sets.py`
根据标注图像与掩膜文件生成训练数据集。

#### `Generate_validing_sets.py`
生成验证数据集。

#### `Generate_testing_sets.py`
生成测试数据集（用于最终评估）。

---

## 模型评估

### `Count_auc.py`

- 计算 ROC 曲线
- 计算 AUC（Area Under the Curve）
- 在检测框层面进行性能评估

---

### `Count_FROC.py`

- 计算 FROC（Free-Response Receiver Operating Characteristic）曲线
- 在不同假阳性率（每体数据）条件下评估检测敏感度

FROC 是医学目标检测任务中的关键评估指标。

---

## 跨切片加权框融合（Cross-Slice Weighted Box Fusion）
### `DBT_WBF.py`

不同于传统的二维检测流程，DBT 数据由三维切片堆叠构成。

本文提出的 DBT_WBF 算法：

- 融合相邻切片中的检测框
- 抑制跨切片的重复预测
- 增强真实病灶的连续响应一致性

该方法有效提升检测稳定性，并改善整体检测性能。
