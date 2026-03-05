# **Point Transformer – Model Implementation**

This folder contains the implementation of the **Point Transformer** model adapted from the official repository:
**Pointcept / PointTransformerV3** ([https://github.com/Pointcept/PointTransformerV3](https://github.com/Pointcept/PointTransformerV3)).

## 📌 Overview

The Point Transformer architecture is designed for point cloud understanding using **vector attention**, **local neighborhood grouping**, and **hierarchical feature extraction**.
It enables learning rich geometric representations directly from unordered 3D points.

## 📂 Contents

* `model.py` — Point Transformer model definition (lightly modified for this project).
* `serialization/` — Utility components required for loading and saving model states.

## ⚠ STM32 Deployment Note

The original model relies on **sparse convolution operations**, which are **not supported by the STM32 NPU**.
Because of this hardware limitation, **full training and evaluation on our radar datasets were not completed**.

## 📘 Reference

For full architecture details, training pipelines, and benchmarks, refer to the official repository linked above.

---
