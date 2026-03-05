# CNN-RES Lightweight People Counting Model

This folder contains a lightweight CNN model with multi-scale residual blocks for people counting from mmWave radar point cloud data with temporal frame stacking.

## Overview

The model combines:
- **Multi-scale CNN layers** for spatial feature extraction at different receptive field sizes
- **Residual connections** for deep network training stability and gradient flow
- **Frame stacking** to capture motion and temporal dynamics
- **Ultra-lightweight architecture** (~60K parameters) optimized for edge deployment

The multi-scale approach processes point clouds with three parallel kernel sizes (1×1, 3×1, 8×1) to capture feature patterns at different spatial scales simultaneously, making it highly efficient for real-time inference.

## Architecture

```
Input: (batch, 1024 points, 1, 5 features) - 8 consecutive frames stacked
   ↓
Stem Conv2D(32, 3×1) → BN → ReLU
   ↓
Residual Block (64 filters):
   Multi-Scale Block:
      ├─ Conv2D(16, 1×1) → BN → ReLU  [Feature mixing]
      ├─ Conv2D(32, 3×1) → BN → ReLU  [Local patterns]
      └─ Conv2D(16, 8×1) → BN → ReLU  [Temporal context]
      Concatenate → Dropout(0.1)
   + Residual Connection → ReLU
   ↓
Residual Block (128 filters):
   Multi-Scale Block:
      ├─ Conv2D(32, 1×1) → BN → ReLU
      ├─ Conv2D(64, 3×1) → BN → ReLU
      └─ Conv2D(32, 8×1) → BN → ReLU
      Concatenate → Dropout(0.15)
   + Residual Connection → ReLU
   ↓
Global Average Pooling
   ↓
Dense(64) + L2(1e-05) + Dropout(0.3)
   ↓
Output: Softmax(num_classes)
```

**Multi-Scale Strategy:**
- 1×1 kernels (25%): Cross-feature interactions
- 3×1 kernels (50%): Local point neighborhood patterns (main branch)
- 8×1 kernels (25%): Wide temporal context across frames

**Model Size:** ~60K parameters (10-15× smaller than CNN-LSTM/CNN-TCN)

## Files

### `cnn_res_dataset_generator.py`
Generates training-ready datasets with temporal frame stacking (identical to CNN-LSTM/CNN-TCN generator).

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

**Note:** The dataset generator outputs 3D arrays `(batch, 1024, 5)`. The model automatically reshapes them to 4D format `(batch, 1024, 1, 5)` for Conv2D processing.

### `cnn_res_model.py`
Complete training pipeline with lightweight multi-scale residual architecture.

**Key Features:**
- Multi-scale convolutional blocks with parallel kernel paths
- Residual connections for gradient stability
- Automatic reshape from 3D to 4D for Conv2D compatibility
- Smart normalization (ignores zero-padded regions)
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
CLASS_WEIGHT_SCALE = 0.5
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

Edit configuration in `cnn_res_dataset_generator.py`:

```python
INPUT_DIR = "split_data"          # From preprocessing stage
OUTPUT_DIR = "processed_dataset"  # Where to save .npy files
FRAMES_TO_STACK = 8               # Temporal window size
POINTS_PER_SAMPLE = 1024          # Total points after stacking
FEATURES = ['x', 'y', 'z', 'velocity', 'snr']
```

Run dataset generation:

```bash
python cnn_res_dataset_generator.py
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

Edit configuration in `cnn_res_model.py` if needed:

```python
DATASET_PATH = 'processed_dataset/'
OUTPUT_DIR = './cnn_res/'
BATCH_SIZE = 16
EPOCHS = 300
USE_SEPARATE_VALIDATION = False  # Set True if you have val split
```

Run training:

```bash
python cnn_res_model.py
```

**Output:**
```
cnn_res/
├── cnn_res_best.h5              # Best model weights
└── cnn_res_metrics.csv          # Evaluation metrics
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

### Regularization Strategy

The model uses multiple regularization techniques:

1. **Spatial Dropout**: 0.1-0.15 rate in residual blocks (drops entire feature maps)
2. **L2 Regularization**: 1e-05 penalty on dense layer weights
3. **Batch Normalization**: Momentum 0.96 after each convolution
4. **Dropout**: 0.3 rate in final dense layer
5. **Class Weighting**: Balances imbalanced datasets (scale=0.5)

### Smart Normalization

Unlike standard normalization, this implementation:
- Identifies valid (non-zero) points using masking
- Computes statistics only from real radar detections
- Preserves zero-padding structure after normalization
- Prevents zero-padding from skewing feature distributions

```python
def normalize_with_mask(data, mean, std):
    valid_mask = np.any(data != 0, axis=-1)
    data_norm = (data - mean) / std
    data_norm = np.where(valid_mask[..., None], data_norm, 0.0)
    return data_norm
```

### Loss Function

**Focal Loss** with gamma=1.0:
- Downweights easy examples to focus on hard cases
- Helps with class imbalance
- Lower gamma (1.0) provides moderate focus on hard examples

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
DROPOUT_RATE = 0.2          # Higher dropout in blocks
FRAMES_TO_STACK = 4         # Fewer frames = more samples
# Reduce to single residual block per stage
```

### For Large Datasets (>10000 samples)
```python
L2_REG = 1e-06              # Lighter regularization
DROPOUT_RATE = 0.05         # Lower dropout
FRAMES_TO_STACK = 16        # More temporal context
# Add more residual blocks: 32→64→64→128→128
```

### For More Temporal Context
```python
FRAMES_TO_STACK = 16        # Double the temporal window
# Adjust kernel sizes: (1,1), (3,1), (16,1)
# Increase 8×1 kernel to 16×1 for wider receptive field
```

### For Real-Time Inference
```python
# Already optimized! (~60K params)
# Further optimization:
POINTS_PER_SAMPLE = 512     # Reduce point count
FRAMES_TO_STACK = 4         # Smaller temporal window
# Use 32→64 channels only (single residual block)
# Expected params: ~20K, inference time: <3ms on edge devices
```

### For Higher Accuracy (Trade-off: Model Size)
```python
# Add deeper architecture: 32→64→64→128→128→256
# Increase dense layer: Dense(128)
# Expected params: ~250-300K
```

## Model Performance Example
<img width="918" height="975" alt="image" src="https://github.com/user-attachments/assets/3ce685da-af84-4710-b6b9-cedc497def2e" />
