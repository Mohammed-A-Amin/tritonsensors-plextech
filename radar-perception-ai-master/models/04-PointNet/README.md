# **PointNet – Radar-Based People Counting Model**

This folder contains an updated implementation of **PointNet**, originally based on the official PyTorch version from:

**[https://github.com/fxia22/pointnet.pytorch/tree/master](https://github.com/fxia22/pointnet.pytorch/tree/master)**

The model has been extensively modified to support **5-dimensional radar point cloud data** and optimized for the **People Counting** task (4-class classification).
All architectural components have been adapted to work with our domain-specific radar features.

---

## **1. Overview**

PointNet is a neural network designed to process **unordered point clouds**.
Unlike traditional convolutional models that rely on structured grids, PointNet directly consumes raw points and performs permutation-invariant feature learning.

In this repository, PointNet is adapted for:

### ✔ Radar-based multi-object and people counting

### ✔ Handling 5D point features:

* **x, y, z** (position)
* **velocity**
* **SNR (Signal-to-Noise Ratio)**

### ✔ Output: 4-class classification

(e.g., no-person, 1-person, 2-people, 3+ people)

This modification allows the model to learn spatial, motion, and signal-strength patterns essential for radar sensing.

---

## **2. Input Format**

### **Expected input shape**

```
(batch_size, num_points, 5)
```

Before being processed by the network, input is transposed to:

```
(batch_size, 5, num_points)
```

This is required because PointNet uses **1D convolutions along the point dimension**.

---

## **3. Architecture Breakdown**

The architecture includes three major components:

---

## **3.1 Spatial Transformer Network (STN5d)**

The original PointNet uses a **T-Net** to learn a 3×3 transformation matrix for (x,y,z).
In our radar version, the input has **5 channels**, so we implement:

### **STN5d**

Learns a **5×5** affine transformation matrix.

#### Purpose:

* Canonical alignment of input points
* Robustness against rotation, scaling, radar noise, and viewpoint variations

#### Structure:

* Conv1d: 5 → 64
* Conv1d: 64 → 128
* Conv1d: 128 → 1024
* Fully connected: 1024 → 512 → 256 → 25
* Output reshaped to (batch_size, 5, 5)
* Identity matrix added for stability

The STN outputs a matrix **T**, and the input point cloud is transformed via:

```
x' = T · x
```

---

## **3.2 Feature Extraction (PointNetfeat)**

This block produces global point cloud features and optionally applies a **feature-transform T-Net**.

### **Main Pipeline**

1. Apply STN5d
2. 1D convolution layers:

   * 5 → 64
   * 64 → 128
   * 128 → 1024
3. BatchNorm + ReLU on each conv layer
4. Global Max Pooling → produces a **1024-dim feature vector**

### **Optional Feature Transform (STNkd)**

* Learns a **64×64** transformation
* Used to stabilize feature learning and enforce orthogonality
* Regularized using `feature_transform_regularizer`

The feature extractor outputs:

```
global_feature_vector (1024), spatial_transform_matrix, feature_transform_matrix
```

---

## **3.3 Classification Head (PeopleCountingPointNet)**

The classifier converts the 1024-dimensional global feature into class probabilities.

### Layers:

* FC: 1024 → 512
* FC: 512 → 256
* FC: 256 → num_classes (4)

### Additional components:

* BatchNorm
* ReLU
* Dropout (p=0.3) for regularization
* No Softmax (because CrossEntropyLoss expects raw logits)

---

## **4. Feature Transform Regularization**

PointNet includes a special regularizer to keep the learned transform matrices close to orthogonal:

```
|| T · Tᵀ − I ||₂
```

This prevents the T-Net from collapsing into degenerate transformations.

Used only when **feature_transform=True**.

---

## **5. Why PointNet Works Well for Radar?**

* Radar point clouds are **unordered**, making PointNet a natural fit
* Velocity and SNR are highly discriminative features
* Global max pooling ensures invariance to point permutation
* T-Net helps mitigate sensor noise, missing points, or viewpoint changes
* Lightweight architecture suitable for embedded platforms

---

## **6. Differences From the Original PointNet Repository**

### **Major Modifications**

✔ Input dimension changed **3 → 5**
✔ New STN5d added for 5D transformation
✔ Classification head tailored for **4 classes**
✔ Radar-specific normalization and pipeline integration
✔ Updated evaluation metrics (precision, recall, F1)
✔ Data shape checking and support for variable number of points

---

## **7. Evaluation Pipeline**

The file includes:

* Accuracy, Precision, Recall, F1-score computation
* Confusion matrix visualization
* GPU-safe evaluation loop
* Numpy-based metrics accumulation

---

## **8. Summary**

This modified PointNet provides a strong baseline for radar-based people counting and other perception tasks.
Its ability to learn directly from raw point clouds, combined with the added support for radar velocity and SNR, makes it an efficient and flexible architecture for embedded radar systems.

The model is compact, interpretable, and deployable compared to heavier architectures such as PointTransformer.

---
