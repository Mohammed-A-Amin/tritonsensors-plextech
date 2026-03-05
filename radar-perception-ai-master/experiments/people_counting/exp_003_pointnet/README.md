# Experiment: PointNet for Radar-Based People Counting

This experiment evaluates a modified **PointNet** architecture for FMCW radar point cloud–based people counting.
PointNet was adapted to process **5-dimensional radar points**:
`[x, y, z, velocity, SNR]`.

The experiment includes training scripts, logs, visualization plots, and the final/best model checkpoints.

---

## Files

* **train.py** – Training and evaluation script
* **Train_PointNet.ipynb** – Notebook version of the experiment
* **logs.txt** – Console logs from training
* **training_curves_pointnet.png** – Train/Test accuracy and loss curves
* **confusion_matrix_pointnet.png** – Final confusion matrix
* **best_people_count_pointnet.pth** – Best checkpoint (highest test accuracy)
* **final_people_count_pointnet.pth** – Model from the last epoch
* **README.md** – This documentation

---

## Dataset Summary

Training set:

```
Data shape:  (11497, 1024, 5)
Labels shape: (11497,)
Classes: 3 → {1 person, 2 people, 3 people}
Class distribution: [   0 3829 3830 3838 ]
```

Test set:

```
Data shape: (2870, 1024, 5)
Labels shape: (2870,)
Class distribution: [  0 955 957 958 ]
```

Each sample contains **1024 radar points**, each with 5 features.

---

## Model Summary

* Architecture: Modified **PointNet** with 5D STN
* Parameters: **3,466,589** (all trainable)
* Classifier: 4-class output (0–3 people), though 0-person class is empty in this dataset
* Feature Transform regularization enabled
* Training device: **CUDA**

---

## Training Results

### **Final Test Performance**

| Metric             | Score      |
| ------------------ | ---------- |
| Accuracy           | **90.38%** |
| Precision          | **90.29%** |
| Recall             | **90.39%** |
| F1-score           | **90.31%** |
| Best Test Accuracy | **90.38%** |

---

### **Per-Class Metrics**

| Class        | Precision | Recall | F1-score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| **1 person** | 0.96      | 1.00   | 0.98     | 955     |
| **2 people** | 0.88      | 0.83   | 0.86     | 957     |
| **3 people** | 0.87      | 0.88   | 0.88     | 958     |

Model demonstrates strong performance, especially for **1-person** and **3-people** categories, with slightly lower recall on **2-people** class due to higher intra-class variation.

---

## Output Artifacts

* **confusion_matrix_pointnet.png**
  Visualizes classification quality across all classes.

* **training_curves_pointnet.png**
  Shows training loss, training accuracy, and test accuracy across epochs.

* **Saved Models:**

  * `best_people_count_pointnet.pth` – best accuracy checkpoint
  * `final_people_count_pointnet.pth` – last epoch model

---

## How to Run

```bash
python train.py
```

Ensure the dataset paths inside `train.py` match your directory structure.

---