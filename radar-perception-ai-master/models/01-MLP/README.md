
# People Counting Using Point Cloud MLP

**Baseline Model — FMCW Radar Point Cloud (4-Class Classification)**

This directory contains the implementation of a **point-cloud-only MLP model** for FMCW radar–based **people counting**.
The model operates directly on per-frame radar point clouds and performs **4-class classification**, where each class corresponds to the number of people detected in the radar field of view.

This baseline establishes a simple yet effective benchmark for later comparison with more advanced architectures such as PointNet, PointNet++, and Point Transformer.

---

## 📌 Overview

The implemented model is a **point-wise MLP encoder** followed by **global max pooling** to generate a fixed-length feature vector for each radar frame. A classifier head then predicts the number of people.

---

## 📁 Input Data Structure

Each sample has dimensions of 64*5, meaning 64 points, and each point has the form:

```
[x, y, z, velocity, intensity]
```

---

## 🧠 Model Architecture

### SubMLP Encoder

A point-wise encoder:

```
5 → 64 → 128 → 256  
BatchNorm + ReLU after each layer
Global Max Pooling (across 64 points)
```

### Classification Head

```
256 → 128 → 64 → 4
ReLU activations
Dropout (0.3)
```

### Output

A probability distribution over 4 classes:

* **0 people**
* **1 person**
* **2 people**
* **3+ people (or class 3)**

---


## 🧩 Notes

* This MLP serves as a **baseline experiment** for radar-based people counting.
* It does *not* use spatial relationships between points — therefore, more advanced models (PointNet, Point Transformer) are expected to outperform it.
* The script is easy to extend for:

  * Different point counts
  * Additional radar features
  * Multi-task training
  * Temporal analysis (tracking / fall detection)

---
