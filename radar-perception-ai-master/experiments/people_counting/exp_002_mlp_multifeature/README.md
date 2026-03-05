# **Multi-Modal Radar Classification (NaN-Safe Experimental Training)**

This directory contains an experimental training script for building a **multi-modal radar classification model** using Point Cloud, Tracks, and Height features.
This version includes enhanced stability, NaN handling, safe data loading, and improved training procedures.

---

## 📌 **Key Features**

### ✔ **Multi-Modal Classification Architecture**

The model includes three independent MLP-based encoders:

| Modality   | Input Dim  | Encoder Output |
| ---------- | ---------- | -------------- |
| PointCloud | 5 features | 256-d vector   |
| Tracks     | 9 features | 128-d vector   |
| Height     | 2 features | 64-d vector    |

The outputs are concatenated and fed into a classifier head that predicts the final class label.

---

### ✔ **NaN & Inf Handling**

Before training begins:

* Dataset files are scanned for **NaN** and **Infinite** values.
* In the dataset loader, all NaNs are automatically replaced with `0.0`.
* Infinite values are clipped to a fixed range (`±1e4`).

This is particularly important for `train_height.npy`, which may contain NaNs.

---

### ✔ **Stability Enhancements**

To ensure more stable training, the script includes:

* **LayerNorm** at encoder inputs
* **BatchNorm + Dropout** inside all encoders
* **Gradient Clipping** (`max_norm = 1.0`)
* **AdamW optimizer** with weight decay
* **ReduceLROnPlateau scheduler**

---

### ✔ **Best Model Saving**

During training, the model with the highest **test accuracy** is saved automatically:

```
best_multimodal_model.pth
```

---

### ✔ **Visualization Tools**

After training, the script generates:

* A **Confusion Matrix**
* A **Train/Test Accuracy Curve**
* An exported **architecture graph** using `torchview`:

```
RadarNet_Architecture.png
```

---

## 📂 **Expected Dataset Structure**

Inside the folder `processed_dataset_multimodal/`:

```
train_pointcloud.npy
train_tracks.npy
train_height.npy
train_labels.npy

test_pointcloud.npy
test_tracks.npy
test_height.npy
test_labels.npy
```

All arrays must be stored as:

* `float32` for features
* `int64` for labels

---

## 🚀 **How to Run**

### 1) Install Dependencies

```bash
pip install torch numpy scikit-learn matplotlib seaborn torchview
```

### 2) Run the Training Script

```bash
python train.py
```

### 3) Output Files

After training, you will obtain:

```
best_multimodal_model.pth
RadarNet_Architecture.png
(visualized plots on screen)
```

---

## 🧠 **Model Architecture Overview**

```
PointCloud Encoder → 256-d
Tracks Encoder     → 128-d
Height Encoder     →  64-d
----------------------------------
Concatenation (448-d)
→ Classifier Head
→ 4 Output Classes
```

Each encoder includes:

* Input LayerNorm
* Several Linear + BatchNorm + ReLU + Dropout layers
* Global Max Pooling

---

## 📈 **Metrics Generated**

The evaluation function reports:

* Accuracy
* Precision (Macro)
* Recall (Macro)
* F1-Score (Macro)
* Confusion Matrix

---

## 🎯 **Purpose of This Script**

This script serves as a **research and experimental baseline** for:

* Early validation of multi-modal MLP architectures
* Testing robustness against NaN/Inf values
* Establishing baseline performance
* Preparing for more advanced models (PointNet, Transformer-based models, temporal fusion, etc.)

---