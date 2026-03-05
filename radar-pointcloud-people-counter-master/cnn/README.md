# CNN Dataset Generator & Model Trainer

This repository contains two Python scripts for building and training a CNN-based people counting system using radar sensor data (FMCW radar point clouds).

## Files Overview

### 1. `cnn_dataset_generator.py`
Flexible dataset generation pipeline for preprocessing radar point cloud data into training-ready NumPy arrays.

### 2. `train_cnn.py`
CNN model training script with advanced callbacks, class imbalance handling, and comprehensive evaluation metrics.

---

## `cnn_dataset_generator.py` - Dataset Generator

### Purpose
Transforms raw radar point cloud CSV files into normalized, padded, and optionally reshaped feature tensors suitable for CNN training.

### Key Features

#### Configuration System
- **Dataclass-based configuration** with preset options
- **Interactive CLI builder** for custom configurations
- **Feature selection**: Choose from `x, y, z, velocity, snr`
- **Flexible normalization**: Per-feature min-max normalization with configurable ranges

#### Data Processing Pipeline
1. **Truncation**: Select top N points by SNR or sort-based method
2. **Hierarchical sorting**: Multi-column spatial sorting (x → y → z)
3. **Zero-padding**: Pad to fixed point count for batch training
4. **Temporal fusion**: Concatenate multiple consecutive frames
5. **Reshaping**: Convert point sequences to HxWxC image-like tensors

#### Configuration Parameters
```python
Config(
    features=['x', 'y', 'z', 'velocity', 'snr'],  # Selected features
    max_points=64,                                 # Fixed point count
    use_zero_pad=True,                             # Pad if < max_points
    truncate_method='highest_snr',                 # 'highest_snr' or 'sort_based'
    sort_by=['x', 'y', 'z'],                       # Hierarchical sort order
    normalize_these=['velocity', 'snr'],           # Features to normalize
    use_reshape=False,                             # Enable HxWxC reshaping
    reshape_h=8, reshape_w=8,                      # Reshape dimensions
    frames_per_sample=1,                           # Temporal window size
    frame_stride=1                                 # Stride for temporal sliding
)
```

### Usage

#### Quick Start (Preset Configuration)
```python
from cnn_dataset_generator import preset_basic, UpdatedRadarGenerator

config = preset_basic()  # 5 features, 64 points, no reshape
generator = UpdatedRadarGenerator(config)
generator.run()
```

#### Interactive Configuration
```python
from cnn_dataset_generator import build_custom_config, UpdatedRadarGenerator

config = build_custom_config()  # Interactive CLI prompts
generator = UpdatedRadarGenerator(config)
generator.run()
```

#### Custom Configuration
```python
from cnn_dataset_generator import Config, UpdatedRadarGenerator

config = Config(
    input_dir='split_data',
    output_dir='processed_dataset',
    features=['x', 'y', 'z', 'velocity', 'snr'],
    max_points=64,
    use_reshape=True,
    reshape_h=8,
    reshape_w=8,
    frames_per_sample=3,
    normalize_these=['velocity', 'snr']
)

generator = UpdatedRadarGenerator(config)
generator.run()
```

### Input Format
Expected directory structure:
```
split_data/
├── experiment1_pointcloud_train.csv
├── experiment1_summary_all.csv
├── experiment2_pointcloud_val.csv
├── experiment2_summary_all.csv
└── ...
```

**Pointcloud CSV columns**: `frame_num`, `x`, `y`, `z`, `velocity`, `snr`
**Summary CSV columns**: `frame_num`, `ground_truth` (people count as string or int)

### Output Format
```
processed_dataset/
├── data_train.npy      # Shape: (N_train, max_points, num_features) or (N, H, W, C)
├── labels_train.npy    # Shape: (N_train,)
├── data_val.npy
├── labels_val.npy
├── data_test.npy
├── labels_test.npy
└── config.txt          # Configuration summary
```

### Advanced Features
- **Reshape validation**: Ensures max_points = H × W when reshaping
- **Missing data handling**: Graceful fallback for missing features/labels

---

## `train_cnn.py` - CNN Model Trainer

### Purpose
Trains a 2D CNN classifier for people counting with imbalanced class handling, early stopping, and comprehensive evaluation.

### Model Architecture

```
Input: (batch, 64, 5) or (batch, H, W, C)
│
├─ Conv2D(16, 3x3, ReLU, same) + Dropout(0.3)
├─ Conv2D(32, 3x3, ReLU, same) + Dropout(0.3)
├─ BatchNormalization
├─ Flatten
├─ Dense(512, ReLU)
├─ BatchNormalization + Dropout(0.4)
└─ Dense(num_classes, Softmax)

Optimizer: Adam(lr=0.001, beta_1=0.5)
Loss: Categorical Crossentropy
```

### Key Features

#### Training Enhancements
- **Class imbalance handling**: Computed class weights for balanced training
- **Early stopping**: Patience of 25 epochs with best weight restoration
- **Learning rate decay**: ReduceLROnPlateau with factor 0.1, patience 10
- **Model checkpointing**: Saves best model based on validation loss

#### Validation Strategy
- **Flexible validation**: Use separate validation set or reuse test set
- **GPU auto-configuration**: Automatic memory growth for TensorFlow

#### Comprehensive Evaluation
Metrics calculated for all data splits (train/val/test/all):
- Accuracy
- F1-score (macro)
- Precision (macro)
- Recall (macro)
- Per-class classification report
- Confusion matrix

### Configuration

```python
# Training parameters
USE_SEPARATE_VALIDATION = False  # Use test set for validation if no val set
BATCH_SIZE = 32
EPOCHS = 300  # Early stopping will terminate earlier

# Paths
DATASET_PATH = './input/'
OUTPUT_DIR = './train_model/'
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'CNN_people_count_best.h5')
METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, 'people_count_final_metrics.csv')
```

### Usage

```bash
python train_cnn.py
```

### Input Files (Expected in DATASET_PATH)
```
data_train.npy       # Training features
labels_train.npy     # Training labels
data_test.npy        # Test features
labels_test.npy      # Test labels
data_val.npy         # Optional validation features
labels_val.npy       # Optional validation labels
```

### Output Files
```
train_model/
├── CNN_people_count_best.h5            # Best trained model
└── people_count_final_metrics.csv       # Evaluation metrics for all splits
```

### Callbacks Used
1. **EarlyStopping**: Monitors `val_loss`, patience=25, restores best weights
2. **ModelCheckpoint**: Saves best model based on `val_loss`
3. **ReduceLROnPlateau**: Reduces learning rate by 0.1 when `val_loss` plateaus (patience=10)

### Evaluation Output
The script prints detailed metrics for each split:
- Overall accuracy, F1-macro, precision, recall
- Per-class precision, recall, F1-score
- Confusion matrix (pandas DataFrame format)

Example CSV output:
```
split,accuracy,f1_macro,precision_macro,recall_macro
train,0.9523,0.9481,0.9502,0.9465
test,0.9124,0.9087,0.9102,0.9071
all,0.9387,0.9342,0.9359,0.9328
```

---

## Complete Workflow

### Step 1: Generate Dataset
```python
# Configure and run dataset generator
from cnn_dataset_generator import Config, UpdatedRadarGenerator

config = Config(
    input_dir='split_data',
    output_dir='processed_dataset',
    features=['x', 'y', 'z', 'velocity', 'snr'],
    max_points=64,
    use_zero_pad=True,
    normalize_these=['velocity', 'snr']
)

generator = UpdatedRadarGenerator(config)
generator.run()
```

### Step 2: Train Model
```bash
# Update DATASET_PATH in train_cnn.py to 'processed_dataset'
python train_cnn.py
```

### Step 3: Evaluate Results
```python
# Load best model and metrics
from keras.models import load_model
import pandas as pd

model = load_model('./train_model/CNN_people_count_best.h5')
metrics = pd.read_csv('./train_model/people_count_final_metrics.csv')
print(metrics)
```

---

## Data Shape Examples

| Configuration | Output Shape | Description |
|---------------|--------------|-------------|
| Default | `(N, 64, 5)` | 64 points, 5 features (x, y, z, velocity, snr) |
| With reshape 8×8 | `(N, 8, 8, 5)` | Image-like 8×8 grid with 5 channels |

---

## Dependencies

### Dataset Generator
```
pandas
numpy
```

### Model Trainer
```
tensorflow>=2.x
keras
scikit-learn
pandas
numpy
matplotlib
```

### Installation
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

---
## Outputs Sample

<img width="750" height="644" alt="image" src="https://github.com/user-attachments/assets/90f29a36-7642-4eaa-8af5-43d4d463df06" />

```
--- Evaluating: TRAIN ---
Overall Metrics:
  - Accuracy:  0.7672
  - F1-Macro:  0.7507
  - Precision: 0.7367
  - Recall:    0.7765
Per-Class Metrics (Classification Report):
              precision    recall  f1-score   support

    0_person       0.86      1.00      0.92      1649
    1_person       0.69      0.66      0.67      1591
    2_person       0.85      0.71      0.77      3320
    3_person       0.56      0.73      0.63       988

    accuracy                           0.77      7548
   macro avg       0.74      0.78      0.75      7548
weighted avg       0.78      0.77      0.77      7548

Confusion Matrix:
          0_person  1_person  2_person  3_person
0_person      1649         0         0         0
1_person       274      1057       193        67
2_person         0       447      2364       509
3_person         5        37       225       721
--- Evaluating: TEST ---
Overall Metrics:
  - Accuracy:  0.6681
  - F1-Macro:  0.6392
  - Precision: 0.6277
  - Recall:    0.6607
Per-Class Metrics (Classification Report):
              precision    recall  f1-score   support

    0_person       0.83      1.00      0.91       704
    1_person       0.51      0.44      0.47       675
    2_person       0.73      0.64      0.68      1409
    3_person       0.44      0.56      0.49       418

    accuracy                           0.67      3206
   macro avg       0.63      0.66      0.64      3206
weighted avg       0.67      0.67      0.66      3206

Confusion Matrix:
          0_person  1_person  2_person  3_person
0_person       704         0         0         0
1_person       145       296       197        37
2_person         0       244       908       257
3_person         0        37       147       234
--- Evaluating: ALL ---
Overall Metrics:
  - Accuracy:  0.7377
  - F1-Macro:  0.7181
  - Precision: 0.7051
  - Recall:    0.7421
Per-Class Metrics (Classification Report):
              precision    recall  f1-score   support

    0_person       0.85      1.00      0.92      2353
    1_person       0.64      0.60      0.62      2266
    2_person       0.81      0.69      0.75      4729
    3_person       0.52      0.68      0.59      1406

    accuracy                           0.74     10754
   macro avg       0.71      0.74      0.72     10754
weighted avg       0.75      0.74      0.74     10754

Confusion Matrix:
          0_person  1_person  2_person  3_person
0_person      2353         0         0         0
1_person       419      1353       390       104
2_person         0       691      3272       766
3_person         5        74       372       955
✓ Final metrics saved to: ./train_model/people_count_final_metrics.csv
================================================================================
FINAL SUMMARY
================================================================================
Final performance of the best model:
split  accuracy  f1_macro  precision_macro  recall_macro
train  0.767223  0.750717         0.736714      0.776542
 test  0.668122  0.639160         0.627658      0.660689
  all  0.737679  0.718129         0.705130      0.742055
Outputs:
  - Best model saved at: ./train_model/MARS_people_count_best.h5
  - Metrics report at:   ./train_model/people_count_final_metrics.csv
✓ Process complete.
```
