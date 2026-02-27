# Enhanced Mass and Architectural Distortion Detection in Digital Breast Tomosynthesis via Cross-Slice Weighted Box Fusion

[**中文版 README**](readme_zh.md)

## Overview

This repository provides the official implementation of our paper:

**“Enhanced Mass and Architectural Distortion Detection in Digital Breast Tomosynthesis via Cross-Slice Weighted Box Fusion”**  
(Under review at *Pattern Recognition*)

We propose a dedicated detection and evaluation framework for **Digital Breast Tomosynthesis (DBT)** that enhances mass and architectural distortion detection by introducing a novel **Cross-Slice Weighted Box Fusion (DBT_WBF)** algorithm.

The framework includes:

- Dataset preparation
- YOLOv8-based model training
- Inference pipeline
- AUC and FROC evaluation
- Cross-slice detection box fusion

Our method leverages the 3D nature of DBT volumes by integrating predictions across adjacent slices to improve robustness and reduce false positives.

---

## Project Structure

### `config.py`
Configuration file for training and inference.

Includes:
- Model parameters
- Training hyperparameters
- Dataset paths
- Inference thresholds

The detector is based on **YOLOv8**, trained using the **MMYOLO / MMDetection** framework:  
https://mmyolo.readthedocs.io/zh-cn/latest/

---

## Dataset Preparation

We use the **BCS-DBT dataset**:  
https://sites.duke.edu/mazurowski/resources/digital-breast-tomosynthesis-database/

### Scripts

#### `Generate_training_sets.py`
Generates the training dataset from annotated images and masks.

#### `Generate_validing_sets.py`
Generates the validation dataset.

#### `Generate_testing_sets.py`
Generates the testing dataset for final evaluation.

---

## Model Evaluation

### `Count_auc.py`
- Computes ROC curves
- Calculates AUC (Area Under the Curve)
- Evaluates detection-box-level performance

### `Count_FROC.py`
- Computes FROC (Free-Response Receiver Operating Characteristic) curves
- Measures sensitivity under varying false positive rates per volume

FROC is particularly important in medical object detection tasks.

---

### Cross-Slice Weighted Box Fusion

Unlike conventional 2D detection pipelines, DBT data consists of 3D slice stacks.  
Our proposed DBT_WBF algorithm:

- Aggregates detection boxes across adjacent slices
- Suppresses redundant cross-slice predictions
- Enhances consistent lesion responses
