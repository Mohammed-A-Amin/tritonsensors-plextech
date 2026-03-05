# Radar People Counting with Random Forest

A machine learning pipeline for classifying people count (0, 1, 2, or 3 people) using radar point cloud data from Texas Instruments mmWave sensors. This project extracts statistical features from radar frames and trains a Random Forest classifier optimized for imbalanced datasets using F1-score.

## Overview

This repository contains two main components:

1. **Feature Extraction** (`rf_dataset_generator.py`): Processes radar point cloud CSV files and extracts 29+ statistical features per frame
2. **Model Training** (`rf_model.py`): Trains and evaluates a Random Forest classifier with hyperparameter tuning

## Features

### Dataset Generator (`rf_dataset_generator.py`)

**Input:** Radar point cloud CSV files with columns:
- `frame_num`, `x`, `y`, `z`, `velocity`, `snr`, `noise`, `track_idx`

**Preprocessing Options:**
- Spatial filtering: Remove points outside room boundaries (configurable x/y/z limits)
- Velocity filtering: Remove static/low-velocity points for denoising
- SNR filtering: Remove low-quality detections

**Extracted Features (29 base features):**
1. `num_total_points` - Total detected points per frame
2. `num_tracked_objects` - Tracked objects from summary (optional)
3-8. Range statistics: mean, std of range, Doppler, SNR
9-20. Spatial statistics: mean, std, min, max of x, y, z coordinates
21-23. Spatial ratios: x/y, x/z, z/y width ratios
24-29. Magnitude features: mean absolute values of range, Doppler, SNR, x, y, z

**Output:** Three CSV files:
- `features_train.csv`
- `features_val.csv`
- `features_test.csv`

### Random Forest Model (`rf_model.py`)

**Features:**
- Multi-class classification (0, 1, 2, 3 people)
- Grid search hyperparameter optimization
- Weighted F1-score optimization for class imbalance
- Automatic train/val split if validation set not provided
- Confusion matrix visualization
- Model serialization with joblib

**Hyperparameter Search:**
- `n_estimators`: [100, 200, 300]
- `max_depth`: [None, 10, 20, 30]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]

## Installation

```bash
# Required packages
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

## Usage

### 1. Generate Feature Dataset

Edit configuration in `rf_dataset_generator.py`:

```python
# Input/Output paths
INPUT_DIR = "split_data"  # Directory with *_pointcloud_train.csv, etc.
OUTPUT_DIR = "classical_ml_dataset"

# Enable/disable preprocessing filters
ENABLE_SPATIAL_FILTER = False
ENABLE_VELOCITY_FILTER = False
ENABLE_SNR_FILTER = False

# Feature options
ADD_TRACKED_OBJECTS_FEATURE = True  # Include num_tracked_objects
ADD_FILTER_STATS = False  # Include filtering statistics
```

Run the feature extractor:

```bash
python rf_dataset_generator.py
```

**Output:**
- `classical_ml_dataset/features_train.csv`
- `classical_ml_dataset/features_val.csv`
- `classical_ml_dataset/features_test.csv`
- `classical_ml_dataset/features_config.txt` (configuration summary)

### 2. Train Random Forest Model

Edit configuration in `rf_model.py`:

```python
DATA_DIR = "classical_ml_dataset"
RANDOM_SEED = 42
TEST_SIZE = 0.1  # Validation split if no val set
N_JOBS = -1  # Use all CPU cores

# Customize hyperparameter grid if needed
PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

Train the model:

```bash
python rf_model.py
```

**Output:**
- Console: Training progress, best parameters, F1 scores, classification report
- `classical_ml_dataset/best_rf_model.pkl` - Trained model
- `classical_ml_dataset/confusion_matrix.png` - Confusion matrix visualization

## Pipeline Workflow

```
Radar Data (CSV)
    ↓
[rf_dataset_generator.py]
    ├── Load point cloud data
    ├── Apply preprocessing filters (optional)
    ├── Extract 29 statistical features per frame
    ├── Merge with ground truth labels
    └── Save train/val/test CSVs
    ↓
Feature CSVs
    ↓
[rf_model.py]
    ├── Load feature datasets
    ├── Grid search hyperparameter tuning (5-fold CV)
    ├── Train Random Forest with best params
    ├── Evaluate on test set
    └── Save trained model
    ↓
Trained Model (best_rf_model.pkl)
```

## File Structure

```
project/
│
├── rf_dataset_generator.py    # Feature extraction script
├── rf_model.py                 # Model training script
│
├── split_data/                 # Input directory (your radar data)
│   ├── recording1_pointcloud_train.csv
│   ├── recording1_summary_all.csv
│   ├── recording2_pointcloud_val.csv
│   └── ...
│
└── classical_ml_dataset/       # Output directory
    ├── features_train.csv
    ├── features_val.csv
    ├── features_test.csv
    ├── features_config.txt
    ├── best_rf_model.pkl
    └── confusion_matrix.png
```

## Configuration

### Spatial Filter Example

```python
ENABLE_SPATIAL_FILTER = True
SPATIAL_LIMITS = {
    'x_min': -3.0,  # meters
    'x_max': 3.0,
    'y_min': 0.0,
    'y_max': 6.0,
    'z_min': -1.0,
    'z_max': 2.5
}
```

### Velocity Filter Example

```python
ENABLE_VELOCITY_FILTER = True
VELOCITY_THRESHOLD = 0.1  # m/s - remove points with |velocity| < 0.1 m/s
```

## Model Evaluation

The model outputs:
- **Weighted F1-score**: Accounts for class imbalance
- **Classification report**: Precision, recall, F1 per class
- **Confusion matrix**: Visual representation of predictions vs. ground truth

Example output:
```
✓ Best params: {'class_weight': 'balanced', 'max_depth': None, 'min_samples_leaf': 3, 'min_samples_split': 10, 'n_estimators': 250}
✓ Best CV f1_weighted: 0.8131
✓ Val f1_weighted: 0.8184

✓ Test f1_weighted: 0.8389

Classification Report:
              precision    recall  f1-score   support

           0       0.95      1.00      0.97       708
           1       0.92      0.80      0.86       685
           2       0.80      0.93      0.86      1429
           3       0.75      0.40      0.52       424

    accuracy                           0.85      3246
   macro avg       0.85      0.78      0.80      3246
weighted avg       0.85      0.85      0.84      3246


✓ Final model F1 (test): 0.8389

✓ Model saved: best_rf_model.pkl in classical_ml_dataset

================================================================================
✓ TRAINING COMPLETE! Use best_rf_model.pkl for predictions.
================================================================================
```
