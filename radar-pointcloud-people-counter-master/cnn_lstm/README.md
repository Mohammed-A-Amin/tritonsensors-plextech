# CNN-LSTM Temporal People Counting Model

This folder contains a hybrid CNN-LSTM deep learning model for people counting from mmWave radar point cloud data with temporal frame stacking.

## Overview

The model combines:
- **CNN layers** for spatial feature extraction from point clouds
- **LSTM layers** for temporal pattern recognition across consecutive frames
- **Frame stacking** to capture motion and temporal dynamics

This architecture is specifically designed for time-series radar data where temporal context significantly improves counting accuracy.

## Architecture

```
Input: (batch, 1024 points, 5 features) - 8 consecutive frames stacked
   ↓
CNN Block 1: Conv1D(128) + BN + SpatialDropout + MaxPool
   ↓
CNN Block 2: Conv1D(64) + BN + SpatialDropout + MaxPool
   ↓
CNN Block 3: Conv1D(64) + BN + SpatialDropout
   ↓
LSTM Block 1: LSTM(64) + BN (return sequences)
   ↓
LSTM Block 2: LSTM(64) + BN (return sequences)
   ↓
Global Average Pooling
   ↓
Dense(64) + BN + Dropout(0.5)
   ↓
Output: Softmax(num_classes)
```

## Files

### `cnn_lstm_dataset_generator.py`
Generates training-ready datasets with temporal frame stacking.

**Key Features:**
- Stacks consecutive frames (default: 8 frames per sample)
- Applies zero-padding to fixed point count (default: 1024 points)
- Feature selection: x, y, z, velocity (configurable)
- Label aggregation from stacked frames (last/max/mean/first)
- Outputs NumPy arrays optimized for TensorFlow/Keras

**Configuration:**
```python
FRAMES_TO_STACK = 8          # Number of consecutive frames
POINTS_PER_SAMPLE = 1024     # Points after stacking all frames
FEATURES = ['x', 'y', 'z', 'velocity', 'snr']
LABEL_AGGREGATION = 'last'   # How to combine labels from frames
```

**Output:**
- `data_train.npy`: Shape (num_samples, 1024, 5)
- `labels_train.npy`: Shape (num_samples,)
- Similar files for val/test splits

### `cnn_lstm_model.py`
Complete training pipeline with regularization and monitoring.

**Key Features:**
- Focal loss for handling class imbalance
- Extensive regularization (L2, Dropout, Batch Normalization)
- Class weighting for imbalanced datasets
- Custom F1-score metric for multi-class evaluation
- Early stopping and learning rate scheduling
- Model checkpointing with best F1-score

**Hyperparameters:**
```python
BATCH_SIZE = 16
EPOCHS = 300
LEARNING_RATE = 0.001
L2_REG = 1e-04
DROPOUT_RATE = 0.3
FOCAL_GAMMA = 2.0
```

## Requirements

```bash
pip install numpy pandas tensorflow keras scikit-learn
```

**Versions tested:**
- TensorFlow >= 2.10
- Python >= 3.8

## Usage

### Step 1: Generate Dataset

First, ensure you have preprocessed CSV files from `data_preprocessing/combined_parser_splitter.py`.

Edit configuration in `cnn_lstm_dataset_generator.py`:

```python
INPUT_DIR = "split_data"          # From preprocessing stage
OUTPUT_DIR = "processed_dataset"  # Where to save .npy files
FRAMES_TO_STACK = 8               # Temporal window size
POINTS_PER_SAMPLE = 1024          # Total points after stacking
FEATURES = ['x', 'y', 'z', 'velocity', 'snr']
```

Run dataset generation:

```bash
python cnn_lstm_dataset_generator.py
```

**Output:**
```
processed_dataset/
├── data_train.npy        # (N_train, 1024, 5)
├── labels_train.npy      # (N_train,)
├── data_test.npy         # (N_test, 1024, 5)
├── labels_test.npy       # (N_test,)
├── data_val.npy          # (N_val, 1024) - if validation split exists
├── labels_val.npy        # (N_val,)
└── dataset_config.txt    # Configuration record
```

### Step 2: Train Model

Edit configuration in `cnn_lstm_model.py` if needed:

```python
DATASET_PATH = 'processed_dataset/'
OUTPUT_DIR = './cnn_lstm_model/'
BATCH_SIZE = 16
EPOCHS = 300
USE_SEPARATE_VALIDATION = False  # Set True if you have val split
```

Run training:

```bash
python cnn_lstm_model.py
```

**Output:**
```
cnn_lstm_model/
├── cnn_lstm_best.h5              # Best model weights
└── cnn_lstm_metrics_final.csv    # Evaluation metrics
```

### Step 3: Evaluate Results

The script automatically evaluates on train and test sets and prints:
- Accuracy and Macro F1-score
- Confusion matrix per class
- Per-class performance metrics

Example output:
```
--- TRAIN ---
Accuracy: 0.9512, Macro F1: 0.9423
Confusion Matrix:
          0_person  1_person  2_person  3_person
0_person       245         3         2         0
1_person         4       512         8         1
2_person         2         6       487         5
3_person         0         2         7       241

--- TEST ---
Accuracy: 0.8876, Macro F1: 0.8654
...
```

## Model Details

### Temporal Frame Stacking

Unlike traditional single-frame approaches, this model stacks **8 consecutive frames** into a single sample:

**Benefits:**
- Captures motion patterns and velocity information
- Provides temporal context for distinguishing stationary vs. moving targets
- Reduces ambiguity in occlusion scenarios
- Improves robustness to temporary detection failures

**Trade-offs:**
- Requires more frames per prediction (8 frames minimum)
- Higher memory usage during training
- Introduces temporal dependencies in data splitting

### Regularization Strategy

The model uses aggressive regularization to prevent overfitting:

1. **Gaussian Noise**: 0.05 std at input (simulates sensor noise)
2. **Spatial Dropout**: 0.3 rate (drops entire feature maps)
3. **L2 Regularization**: 1e-04 penalty on weights
4. **Batch Normalization**: Momentum 0.96
5. **Dropout**: 0.5 rate in final dense layer
6. **Class Weighting**: Balances imbalanced datasets

### Loss Function

**Focal Loss** with gamma=2.0:
- Focuses training on hard-to-classify examples
- Reduces the relative loss for well-classified examples
- Helps with class imbalance without extreme weight adjustments

Formula: `FL(p_t) = -α(1-p_t)^γ log(p_t)`

### Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Macro F1-Score**: Average F1 across all classes (treats each class equally)
- **Confusion Matrix**: Per-class performance breakdown

Macro F1 is the primary metric for model selection as it accounts for class imbalance.

## Hyperparameter Tuning Guide

### For Small Datasets (<1000 samples)
```python
L2_REG = 1e-03              # Stronger regularization
DROPOUT_RATE = 0.4          # Higher dropout
FRAMES_TO_STACK = 4         # Fewer frames = more samples
```

### For Large Datasets (>10000 samples)
```python
L2_REG = 1e-05              # Lighter regularization
DROPOUT_RATE = 0.2          # Lower dropout
FRAMES_TO_STACK = 16        # More temporal context
```

### For Imbalanced Classes
```python
CLASS_WEIGHT_SCALE = 1.0    # Full class weighting
FOCAL_GAMMA = 3.0           # Higher focus on hard examples
```
## Model Performance Example
<img width="861" height="915" alt="image" src="https://github.com/user-attachments/assets/4c243cc8-3d0c-4d76-a2cf-ef569a73bd73" />
