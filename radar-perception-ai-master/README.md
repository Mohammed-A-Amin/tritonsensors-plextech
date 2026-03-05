# Radar Perception AI

**FMCW Radar-Based Human Understanding Using Machine Learning**

---

## Overview

This repository contains a long-term industrial research and development project focused on **human perception using FMCW radar data** and **machine learning models**.

The goal of this project is to design, train, evaluate, and compare a wide range of AI models — from simple baselines to advanced point-based and temporal architectures — for extracting high-level human-related information from radar point cloud data.

The project is designed as a **evolving system**, where:

* Radar datasets are collected progressively
* Models are trained iteratively as new data becomes available
* Experimental results are versioned, reproducible, and comparable

---

## Target Capabilities (Tasks)

The following perception tasks are addressed in this repository:

* **People Counting**
  Estimating the number of people present in the radar field of view.

* **People Tracking**
  Tracking individuals over time using radar point clouds and temporal association.

* **Fall Detection**
  Detecting fall events using spatial and temporal radar patterns.

* **Lack of Movement Detection**
  Identifying prolonged inactivity or abnormal motion absence.

Each task is treated as an **independent problem definition**, with its own evaluation metrics and post-processing logic.

---

## Design Philosophy

This repository follows a **task-centric and experiment-driven** design:

* Tasks are defined independently of model architectures
* Models are reusable across multiple tasks
* Every trained model corresponds to a clearly defined experiment
* Results are stored and documented for long-term comparison
* Dataset versions are explicitly tracked

This structure prevents experiment overwriting, enables scientific comparison, and supports long-term maintainability.

---

## Repository Structure

```text
radar-perception-ai/
│
├── README.md
├── requirements.txt / pyproject.toml
│
├── datasets/                 # Radar datasets (versioned)
│   ├── raw/                  # Original raw radar data
│   ├── processed/            # Preprocessed point cloud data
│   └── annotations/          # Task-specific annotations
│
├── tasks/                    # Task definitions and evaluation logic
│   ├── people_counting/
│   ├── people_tracking/
│   ├── fall_detection/
│   └── lack_of_movement/
│
├── models/                   # Reusable model architectures
│   ├── mlp/
│   ├── pointnet/
│   ├── point_transformer/
│   └── temporal/
│
├── experiments/              # All training experiments (core of the project)
│   ├── people_counting/
│   │   ├── exp_001_mlp_baseline/
│   │   ├── exp_002_pointnet/
│   │   └── exp_003_point_transformer/
│   └── ...
│
├── evaluation/               # Cross-experiment analysis and comparison
│   ├── tables/
│   └── plots/
│
├── configs/                  # Dataset, model, and task configurations
│
├── tools/                    # Utility scripts (visualization, parsing, etc.)
└── docs/                     # Technical documentation and guidelines
```

---

## Radar Data Representation

Radar data is represented as **point cloud frames**, where each frame consists of a fixed number of points.

Each point is encoded as:

```text
[x, y, z, velocity, intensity]
```

Typical tensor shapes:

| Data         | Shape                    |
| ------------ | ------------------------ |
| Single frame | (N_points, 5)            |
| Dataset      | (N_samples, N_points, 5) |

Point cloud normalization, filtering, and temporal aggregation are handled in the dataset preprocessing pipeline.

---

## Current Baseline: People Counting with Point Cloud MLP

The current baseline implementation focuses on **people counting** using a point-wise MLP architecture.

### Model Overview

* Input: Radar point cloud frame `(N_points, 5)`
* Point-wise MLP encoder
* Global max pooling across points
* Classification head

**Example architecture:**

* Encoder: 64 → 128 → 256 (with BatchNorm + ReLU)
* Global Max Pooling
* Classifier: 256 → 128 → 64 → C classes

The task is currently formulated as a **multi-class classification problem**, where each class represents a specific number of people.

---

## Experiments and Results

Each trained model is stored as a separate experiment with:

* Configuration file
* Training script
* Saved weights
* Quantitative results
* Notes and observations

Example:

```text
experiments/people_counting/exp_001_mlp_baseline/
├── train.py
├── config.yaml
├── results.json
└── notes.md
```

### Evaluation Metrics

Depending on the task, evaluation may include:

* Accuracy
* Precision / Recall / F1-score (macro)
* Confusion Matrix
* Temporal consistency metrics (for tracking and activity detection)

---

## Running an Experiment

Example (baseline people counting):

```bash
cd experiments/people_counting/exp_001_mlp_baseline
python train.py
```

All outputs (models, logs, plots) are stored inside the corresponding experiment folder.

---

## Dataset Evolution and Versioning

* Raw data is never modified
* Processed datasets are versioned
* Annotations are task-specific
* Large files may be managed via Git LFS or external storage

This ensures reproducibility across time as the dataset grows.

---

## Future Work

Planned extensions include:

* Advanced point-based architectures (PointNet++, Point Transformer)
* Temporal models (LSTM, Temporal Transformer)
* Multi-task learning across perception tasks
* Real-time inference optimization
* Embedded and edge deployment experiments

---

## Notes

This repository is intended to serve as:

* A long-term industrial perception project
* A research platform for radar-based human understanding
* A reproducible benchmark for model comparison

---