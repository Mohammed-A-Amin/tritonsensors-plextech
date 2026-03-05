# Multi-Modal Radar Classification Model

### Point Cloud + Tracks + Height Fusion using MLP Encoders

This directory contains the implementation of a **multi-modal deep learning model** designed for radar-based classification tasks.
The network processes three different radar feature streams:

* **Point Cloud data** (x, y, z, velocity, snr)
* **Object Tracks** (position, velocity, acceleration)
* **Height features** (height, z)

Each branch is encoded separately using an enhanced **SubMLP encoder** with LayerNorm, BatchNorm, Dropout, and Global Max Pooling.
The encoded features are then fused for final classification.

This architecture is designed for radar perception tasks such as:

* **People Counting**
* People Tracking
* Fall Detection
* Motion / No-Motion Classification

---

## 📌 Key Features

* **Three independent MLP branches** for different radar modalities
* **LayerNorm on raw inputs**, effective for unnormalized sensor data
* **BatchNorm + Dropout** for stable and robust training
* **Global Max Pooling** to obtain fixed-dimensional frame-level features
* **Feature-Level Fusion (Concatenation)**
* Final **classification head** for 4-class output
* Built-in `evaluate()` function returning accuracy, precision, recall, F1-score

---

## 🧱 Model Architecture

### **1. Point Cloud Encoder**

Input dimension: **5**
Features: `[x, y, z, velocity, snr]`

Layers:

```
LayerNorm → [64 → 128] → BatchNorm → ReLU → Dropout → Linear → Global Max Pool
Output: 256-d feature vector
```

---

### **2. Tracks Encoder**

Input dimension: **9**
Features: `[x, y, z, vx, vy, vz, ax, ay, az]`

Layers:

```
LayerNorm → [64 → 128] → BatchNorm → ReLU → Dropout → Linear → Global Max Pool
Output: 128-d feature vector
```

---

### **3. Height Encoder**

Input dimension: **2**
Features: `[height, z]`

Layers:

```
LayerNorm → [32] → BatchNorm → ReLU → Dropout → Linear → Global Max Pool
Output: 64-d feature vector
```

---

### **Fusion + Classification Head**

The three branches are concatenated:

```
256 (pc) + 128 (tracks) + 64 (height) = 448 features
```

Classifier:

```
Linear(448 → 256)
BatchNorm → ReLU → Dropout(0.4)
Linear(256 → 128)
ReLU
Linear(128 → num_classes=4)
```

---

## 📁 Expected Input Format (Data Loader)

Your `DataLoader` should return batches in the form:

```python
(inputs, labels)
where:
    inputs = (pc_batch, tracks_batch, height_batch)
    pc_batch:     (B, N_points, 5)
    tracks_batch: (B, N_tracks, 9)
    height_batch: (B, N_height_points, 2)
```

This design allows flexibility in the number of points (N).

---

## 🚀 Forward Pass Example

```python
model = MultiModalRadarNet(num_classes=4)
logits = model(pc, tracks, height)
```

---

## 📊 Evaluation

You can evaluate any trained model using:

```python
acc, prec, rec, f1, y_true, y_pred = evaluate(model, loader, device)
```

Metrics:

* **Accuracy**
* **Precision (macro)**
* **Recall (macro)**
* **F1-score (macro)**

These values can be logged or saved into JSON for experiment tracking.

---

## ✔️ Applications

This multi-modal architecture is suitable for:

* People Counting (4 classes)
* Human Activity / Fall Detection
* Radar-Based Gesture or Motion Classification
* Multi-sensor fusion experiments


---

## 📝 Notes

* This is a **mid-level complexity model** designed to outperform a simple point-only MLP baseline.
* It is intentionally modular to allow additional branches (e.g., Doppler heatmap, micro-Doppler, temporal sequences).
* For long-term project scaling, this experiment can serve as a **reference template** for future multi-modal architectures.

---
