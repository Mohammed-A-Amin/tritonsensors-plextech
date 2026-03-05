# CNN-TCN Temporal People Counting Model

This folder contains a hybrid CNN-TCN deep learning model for people counting from mmWave radar point cloud data with temporal frame stacking.

## Overview

The model combines:
- **CNN layers** for spatial feature extraction from point clouds
- **TCN (Temporal Convolutional Network)** for temporal pattern recognition with dilated causal convolutions
- **Frame stacking** to capture motion and temporal dynamics
- **Residual connections** for deep network training stability

TCN offers advantages over LSTM including faster training (parallelizable), larger receptive fields with fewer parameters, and better gradient flow through residual connections.

## Architecture

```
Input: (batch, 1024 points, 5 features) - 8 consecutive frames stacked
   ↓
Initial Conv1D(64) - Feature adaptation
   ↓
TCN Block (dilation=1):  Dilated Conv → BN → ReLU → Dropout (×2) + Residual
   ↓
TCN Block (dilation=2):  Dilated Conv → BN → ReLU → Dropout (×2) + Residual
   ↓
TCN Block (dilation=4):  Dilated Conv → BN → ReLU → Dropout (×2) + Residual
   ↓
TCN Block (dilation=8):  Dilated Conv → BN → ReLU → Dropout (×2) + Residual
   ↓
Global Average Pooling
   ↓
Dense(128) + BN + Dropout(0.4)
   ↓
Output: Softmax(num_classes)
```

**Receptive Field Calculation:**
- With kernel_size=3 and dilations [1, 2, 4, 8]
- Each block doubles the receptive field
- Total receptive field ≈ 31 time steps (covers all 8 stacked frames)

## Files

### `cnn_tcn_dataset_generator.py`
Generates training-ready datasets with temporal frame stacking (identical to CNN-LSTM generator).

**Key Features:**
- Stacks consecutive frames (default: 8 frames per sample)
- Applies zero-padding to fixed point count (default: 1024 points)
- Feature selection: x, y, z, velocity, snr (configurable)
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

### `cnn_tcn_model.py`
Complete training pipeline with TCN architecture and regularization.

**Key Features:**
- Temporal Convolutional Network with causal padding (no future information leakage)
- Dilated convolutions for exponential receptive field growth
- Residual connections for training stability
- Focal loss for handling class imbalance
- Custom F1-score metric for multi-class evaluation
- Early stopping and learning rate scheduling
- Model checkpointing with best F1-score

**Hyperparameters:**
```python
BATCH_SIZE = 16
EPOCHS = 300
LEARNING_RATE = 0.001
L2_REG = 1e-05
BN_MOMENTUM = 0.96
FOCAL_GAMMA = 1.0
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

Edit configuration in `cnn_tcn_dataset_generator.py`:

```python
INPUT_DIR = "split_data"          # From preprocessing stage
OUTPUT_DIR = "processed_dataset"  # Where to save .npy files
FRAMES_TO_STACK = 8               # Temporal window size
POINTS_PER_SAMPLE = 1024          # Total points after stacking
FEATURES = ['x', 'y', 'z', 'velocity', 'snr']
```

Run dataset generation:

```bash
python cnn_tcn_dataset_generator.py
```

**Output:**
```
processed_dataset/
├── data_train.npy        # (N_train, 1024, 5)
├── labels_train.npy      # (N_train,)
├── data_test.npy         # (N_test, 1024, 5)
├── labels_test.npy       # (N_test,)
├── data_val.npy          # (N_val, 1024, 5) - if validation split exists
├── labels_val.npy        # (N_val,)
└── dataset_config.txt    # Configuration record
```

### Step 2: Train Model

Edit configuration in `cnn_tcn_model.py` if needed:

```python
DATASET_PATH = 'processed_dataset/'
OUTPUT_DIR = './cnn_tcn_model/'
BATCH_SIZE = 16
EPOCHS = 300
USE_SEPARATE_VALIDATION = False  # Set True if you have val split
```

Run training:

```bash
python cnn_tcn_model.py
```

**Output:**
```
cnn_tcn_model/
├── cnn_tcn_best.h5              # Best model weights
└── cnn_tcn_metrics_final.csv    # Evaluation metrics
```

### Step 3: Evaluate Results

The script automatically evaluates on train and test sets and prints:
- Accuracy and Macro F1-score
- Confusion matrix per class
- Per-class performance metrics

Example output:
```
--- TRAIN_ORIGINAL ---
Accuracy: 0.9587, Macro F1: 0.9501
Confusion Matrix:
          0_person  1_person  2_person  3_person
0_person       248         2         0         0
1_person         3       518         4         0
2_person         1         5       491         3
3_person         0         1         5       244

--- TEST ---
Accuracy: 0.8923, Macro F1: 0.8712
...
```

## Model Details

### Temporal Frame Stacking

Like CNN-LSTM, this model stacks **8 consecutive frames** into a single sample:

**Benefits:**
- Captures motion patterns and velocity dynamics
- Provides temporal context for distinguishing scenarios
- Reduces ambiguity in occlusion and overlap cases
- TCN's causal convolutions naturally respect temporal order

**Trade-offs:**
- Requires minimum 8 frames for inference
- Higher memory during training
- Temporal dependencies in data

### Regularization Strategy

The model uses multiple regularization techniques:

1. **Spatial Dropout**: 0.1 rate in TCN blocks (drops entire feature maps)
2. **L2 Regularization**: 1e-05 penalty on dense layer weights
3. **Batch Normalization**: Momentum 0.96 after each convolution
4. **Dropout**: 0.4 rate in final dense layer
5. **Class Weighting**: Balances imbalanced datasets (scale=0.5)

### Loss Function

**Focal Loss** with gamma=1.0:
- Downweights easy examples to focus on hard cases
- Helps with class imbalance
- Lower gamma (1.0) compared to CNN-LSTM (2.0) as TCN is already robust

Formula: `FL(p_t) = -α(1-p_t)^γ log(p_t)`

### Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Macro F1-Score**: Average F1 across all classes (primary metric)
- **Confusion Matrix**: Per-class performance breakdown

Macro F1 is used for model selection as it accounts for class imbalance.

## Hyperparameter Tuning Guide

### For Small Datasets (<1000 samples)
```python
L2_REG = 1e-04              # Stronger regularization
DROPOUT_RATE = 0.15         # Higher dropout in TCN
FRAMES_TO_STACK = 4         # Fewer frames = more samples
dilations = [1, 2, 4]       # Fewer TCN blocks
```

### For Large Datasets (>10000 samples)
```python
L2_REG = 1e-06              # Lighter regularization
DROPOUT_RATE = 0.05         # Lower dropout
FRAMES_TO_STACK = 16        # More temporal context
dilations = [1, 2, 4, 8, 16, 32]  # Deeper network
```

### For Very Long Sequences
```python
dilations = [1, 2, 4, 8, 16, 32, 64]  # Larger receptive field
num_filters = 64            # Reduce filters to save memory
kernel_size = 5             # Larger kernels
```

### For Real-Time Inference
```python
num_filters = 64            # Reduce filters
dilations = [1, 2, 4]       # Fewer blocks
POINTS_PER_SAMPLE = 512     # Reduce point count
FRAMES_TO_STACK = 4         # Smaller temporal window
```
## Model Performance Example
<img width="863" height="915" alt="image" src="https://github.com/user-attachments/assets/7cd81f09-fd60-4e78-bc39-2f1ee63f5615" />
