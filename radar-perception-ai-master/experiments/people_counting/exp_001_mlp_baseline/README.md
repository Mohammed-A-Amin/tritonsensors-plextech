# People Counting Using Point Cloud MLP

**Baseline Model — FMCW Radar Point Cloud (4-Class Classification)**

This directory contains the implementation of a **point-cloud-only MLP model** for FMCW radar–based **people counting**.
The model operates directly on per-frame radar point clouds and performs **4-class classification**, where each class corresponds to the number of people detected in the radar field of view.

This baseline establishes a simple yet effective benchmark for later comparison with more advanced architectures such as PointNet, PointNet++, and Point Transformer.

---

## 📌 Overview

The implemented model is a **point-wise MLP encoder** followed by **global max pooling** to generate a fixed-length feature vector for each radar frame. A classifier head then predicts the number of people.

Pipeline summary:

1. Load radar point cloud dataset
2. Encode each point independently using an MLP
3. Aggregate point features via max pooling
4. Classify the frame into one of 4 classes
5. Evaluate using accuracy, precision, recall, F1-score
6. Visualize training curves and confusion matrix
7. Save final trained model

---

## 📁 Dataset Structure

The model expects **preprocessed NumPy files**:

| File                     | Shape            | Description                          |
| ------------------------ | ---------------- | ------------------------------------ |
| `radar_data_train.npy`   | (N_train, 64, 5) | Radar point cloud frames (train set) |
| `radar_labels_train.npy` | (N_train,)       | Integer labels (0–3)                 |
| `radar_data_test.npy`    | (N_test, 64, 5)  | Radar point cloud frames (test set)  |
| `radar_labels_test.npy`  | (N_test,)        | Test labels                          |

Each point has the form:

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

## 🚀 Training

To train the model:

```bash
python your_script_name.py
```

The script:

* Loads train/test datasets
* Trains for 30 epochs
* Uses AdamW optimizer + Cosine Annealing LR scheduler
* Logs accuracy per epoch
* Plots training curves
* Saves final weights as:

```
people_count_mlp_point_only_final.pth
```

---

## 📊 Evaluation Metrics

The script computes:

* **Accuracy**
* **Precision (macro)**
* **Recall (macro)**
* **F1-score (macro)**

Additionally, it generates:

* **Confusion matrix**
* **Train vs Test accuracy curve (overfitting check)**

These plots help benchmark the model and compare future architectures.

---

## 📈 Visualization Outputs

The following figures are produced automatically:

1. **Confusion Matrix**

Shows true vs predicted class distribution.

2. **Accuracy Curve**

Useful for diagnosing overfitting or underfitting.

---

## 💾 Saving the Model

At the end of training:

```bash
people_count_mlp_point_only_final.pth
```

is saved in the current directory.

You can later load it via:

```python
model = PeopleCountingPointMLP()
model.load_state_dict(torch.load("people_count_mlp_point_only_final.pth"))
```

---

## 📎 File Structure

```
people_counting/
└── exp_001_mlp_baseline/
    ├── train_mlp.py
    ├── radar_dataset.py
    ├── people_count_mlp.py
    ├── results.json
    ├── confusion_matrix.png
    ├── accuracy_curve.png
    ├── notes.md
    └── people_count_mlp_point_only_final.pth
```

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