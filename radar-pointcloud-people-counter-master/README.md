# Radar Point Cloud People Counter

A comprehensive collection of machine learning and signal processing methods for counting people using mmWave radar point clouds. This repository provides multiple approaches ranging from traditional clustering algorithms to deep learning architectures for real-time occupancy estimation from ceiling-mounted TI mmWave radar sensors.

## Overview

This repository implements various methods for people counting from mmWave radar point cloud data:
- **Deep Learning Models**: CNN, CNN-LSTM, CNN-TCN, CNN-RES for temporal and spatial pattern recognition
- **Classical ML**: Random Forest for feature-based classification
- **Clustering Methods**: DBSCAN with auto-tuning for density-based people detection
- **Signal Processing**: Temporal smoothing for improved radar response stability
- **Data Representation**: Point cloud to 2D heatmap conversion for CNN processing
- **Ensemble Inference**: Production-ready deployment combining multiple models

All methods are designed for **TI IWR6843** or similar mmWave radar sensors mounted on ceilings for indoor people counting applications.

## Key Features

- **Multiple Detection Approaches**: 6+ different methods with unique strengths
- **Temporal Modeling**: Frame stacking and sequential processing for motion patterns
- **Flexible Configuration**: Easy parameter tuning for different environments
- **Comprehensive Metrics**: Accuracy, F1-score, confusion matrices, MDR, FDR, GFR
- **Complete Pipeline**: Data preprocessing → Model-specific generators → Training → Evaluation
- **Advanced Smoothing**: Advanced smoothing algorithms like HMM-based Viterbi for temporal consistency
- **Production Ready**: Optimized for real-time inference with GPU acceleration

## 📁 Repository Structure

```
radar-pointcloud-people-counter/
│
├── 📂 data_preprocessing/              # Core data processing pipeline
│   ├── combined_parser_splitter.py    # JSON to CSV parser & train/test splitter
│   └── README.md                       # Preprocessing documentation
│
├── Deep Learning Models (Temporal)
│   ├── 📂 cnn_heatmap/                 # Frame-stacked temporal CNN (lightweight)
│   │   ├── train_cnn_heatmap.py
│   │   └── README.md
│   │
│   ├── 📂 cnn_lstm/                    # CNN-LSTM hybrid architecture
│   │   ├── cnn_lstm_dataset_generator.py
│   │   ├── cnn_lstm_model.py
│   │   └── README.md
│   │
│   ├── 📂 cnn_tcn/                     # CNN-TCN with dilated convolutions
│   │   ├── cnn_tcn_dataset_generator.py
│   │   ├── cnn_tcn_model.py
│   │   └── README.md
│   │
│   ├── 📂 cnn_res/                     # CNN-RES
│   │   ├── cnn_res_dataset_generator.py
│   │   ├── cnn_tcn_model.py
│   │   └── README.md
│   │
│   └── 📂 cnn/                         # CNN-only architecture
│       ├── cnn_dataset_generator.py
│       ├── train_cnn.py
│       └── README.md
│
├── 📂 ensemble_inference/              # Production inference engine
│   ├── inference_engine.py            # Ensemble model deployment
│   └── README.md                       # Deployment documentation
│
├── 📂 ensemble_viterbi_inference/      # Production inference engine with Viterbi smoothing
│   ├── ensemble_viterbi_inference.py
│   └── README.md
│
├── Classical ML Models
│   └── 📂 random_forest/               # Random Forest classifier
│       ├── rf_dataset_generator.py
│       ├── rf_model.py
│       └── README.md
│
├── Clustering Methods
│   └── 📂 dbscan/                      # DBSCAN with auto-tuning
│       ├── dbscan_people_counter.py
│       └── README.md
│
├── Utilities & Preprocessing
│   ├── 📂 smoothing/                   # Temporal smoothing filters
│   │   ├── radar_temporal_smoothing.py
│   │   └── README.md
│   │
│   ├── 📂 advanced_smoothing/          # Advanced smoothing algorithms
│   │   ├── viterbi/                    # Viterbi temporal consistency
│   │   │   ├── viterbi_realtime.py
│   │   │   └── README.md
│   │   └── README.md
│   │
│   └── 📂 heatmap_dataset_generator/   # Point cloud to 2D image conversion
│       ├── radar_heatmap_dataset_generator.py
│       └── README.md
│
└── README.md                           # This file
```

### Data Preprocessing

All models require preprocessed CSV files from raw JSON radar data:

```bash
cd data_preprocessing

# Configure input files in combined_parser_splitter.py
# Set your JSON files and ground truth labels

python combined_parser_splitter.py
```
**Output:**
- `parsed_data/` - Converted CSV files (pointcloud, tracks, summary)
- `split_data/` - Train/validation/test splits

See [data_preprocessing/README.md](data_preprocessing/README.md) for detailed configuration.

## Production Deployment

For production use, the ensemble inference engine combines CNN-RES and CNN-TCN models:

```bash
cd ensemble_inference

# Run inference on radar JSON files
python inference_engine.py ../data/your_radar_data.json

# With custom models
python inference_engine.py data.json ../cnn_res/model.h5 ../cnn_tcn/model.h5
```

Features:

Real-time ensemble prediction (CNN-RES + CNN-TCN averaging)

Temporal smoothing for stable output

Frame stacking with 8-frame history

Console output with confidence scores

See [ensemble_inference/README.md](ensemble_inference/README.md) for detailed usage.

## Documentation

Each folder contains detailed documentation:

- **[data_preprocessing/README.md](data_preprocessing/README.md)** - Data pipeline and CSV generation
- **[cnn_heatmap/README.md](cnn_heatmap/README.md)** - Frame-stacked temporal CNN architecture and training
- **[cnn_lstm/README.md](cnn_lstm/README.md)** - CNN-LSTM architecture and training
- **[cnn_tcn/README.md](cnn_tcn/README.md)** - TCN architecture and dilated convolutions
- **[cnn_res/README.md](cnn_res/README.md)** - Residual architecture
- **[cnn/README.md](cnn/README.md)** - CNN architecture and training
- **[random_forest/README.md](random_forest/README.md)** - Feature engineering and RF training
- **[dbscan/README.md](dbscan/README.md)** - Clustering parameters and auto-tuning
- **[smoothing/README.md](smoothing/README.md)** - Temporal filtering methods
- **[advanced_smoothing/README.md](advanced_smoothing/README.md)** - Advanced smoothing like HMM-based Viterbi for temporal consistency
- **[heatmap_dataset_generator/README.md](heatmap_dataset_generator/README.md)** - 2D heatmap generation
- **[ensemble_inference/README.md](ensemble_inference/README.md)** - Production deployment and inference
- **[ensemble_viterbi_inference/README.md](ensemble_viterbi_inference/README.md)** - Production deployment and inference with Viterbi smoothing
