# Radar Heatmap Dataset Generator

A Python tool for generating fixed-resolution heatmap image datasets from radar point cloud data, designed for machine learning applications in people detection and tracking.

## Overview

This tool converts radar point cloud data (X, Z coordinates with velocity, SNR features) into fixed-resolution heatmap images with ground truth labels. It supports multiple frame aggregation modes, configurable spatial filtering, and flexible intensity mapping for deep learning model training.

## Features

- **Fixed-Resolution Heatmaps**: Generate consistent 64×64, 128×128, or custom resolution images
- **Configurable Intensity Features**: Use SNR or velocity as heatmap values
- **Spatial Filtering**: 
  - Separate bounding box filters for point cloud preprocessing
  - Independent image spatial limits for visualization
  - Velocity-based filtering (static point removal)
- **Frame Aggregation Modes**:
  - `concat`: Concatenate point clouds from multiple frames
  - `mean`: Average heatmap intensities across frames
  - `sum`: Sum heatmap intensities across frames
  - `max`: Maximum intensity projection across frames
- **Flexible Binning Statistics**: mean, sum, max, min, count, median
- **Multiple Output Formats**: PNG/JPG images, optional NumPy arrays (.npy)
- **Ground Truth CSV**: Automatic generation of metadata with labels and frame info
- **Train/Val/Test Split Support**: Automatic detection from input filenames

## Installation

### Requirements

```bash
pip install pandas numpy matplotlib scipy
```

### Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- scipy

## Usage

### Basic Usage

1. **Prepare Your Data**: Organize parsed radar data in the input directory with:
   - `*_pointcloud.csv`: Point cloud data (columns: frame_num, x, z, velocity, snr)
   - `*_summary.csv`: Ground truth labels (columns: frame_num, ground_truth)

2. **Configure Parameters**: Edit the `main()` function in the script:

```python
config = HeatmapConfig(
    input_dir="parsed_data",
    output_dir="heatmap_dataset",
    resolution=64,
    image_x_limits=(-6.0, 6.0),
    image_z_limits=(-6.0, 6.0),
    intensity_feature="snr",
    aggregate_frames=False
)
```

3. **Run the Generator**:

```bash
python radar_heatmap_dataset_generator.py
```

### Configuration Options

#### Spatial Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | int | 64 | Grid size (N×N pixels) |
| `image_x_limits` | tuple | (-6.0, 6.0) | X-axis range for visualization (meters) |
| `image_z_limits` | tuple | (-6.0, 6.0) | Z-axis range for visualization (meters) |
| `bbox_x_limits` | tuple/None | (-8.0, 8.0) | Point filter X range (None=use image_x_limits) |
| `bbox_z_limits` | tuple/None | (-8.0, 8.0) | Point filter Z range (None=use image_z_limits) |
| `apply_bounding_box` | bool | True | Enable spatial point filtering |

#### Intensity Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `intensity_feature` | str | "snr" | Feature for heatmap intensity: 'snr', 'velocity', 'intensity' |
| `use_abs_velocity` | bool | True | Use absolute velocity value |
| `bin_statistic` | str | "mean" | Binning method: 'mean', 'sum', 'max', 'min', 'count', 'median' |
| `min_velocity` | float/None | 0.1 | Minimum velocity threshold (None=disabled) |

#### Frame Aggregation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aggregate_frames` | bool | False | Enable multi-frame aggregation |
| `frames_per_sample` | int | 1 | Number of consecutive frames per sample |
| `frame_stride` | int | 1 | Stride between aggregated samples |
| `aggregation_mode` | str | "concat" | Aggregation method: 'concat', 'mean', 'sum', 'max' |

#### Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | "heatmap_dataset" | Output directory path |
| `prefix` | str | "radar" | Output filename prefix |
| `suffix_mode` | str | "label" | Suffix type: 'label' (person count) or 'split' (train/val/test) |
| `save_images` | bool | True | Save PNG/JPG images |
| `save_numpy` | bool | False | Save NumPy arrays (.npy) |
| `image_format` | str | "png" | Image format: 'png' or 'jpg' |

### Example Configurations

#### Single Frame Mode (Default)

```python
config = HeatmapConfig(
    input_dir="parsed_data",
    output_dir="heatmap_64x64_snr",
    resolution=64,
    intensity_feature="snr",
    aggregate_frames=False
)
```

#### Multi-Frame Concatenation Mode

```python
config = HeatmapConfig(
    input_dir="parsed_data",
    output_dir="heatmap_128x128_8frames",
    resolution=128,
    aggregate_frames=True,
    frames_per_sample=8,
    aggregation_mode="concat",
    intensity_feature="velocity"
)
```

#### High-Resolution with Maximum Projection

```python
config = HeatmapConfig(
    input_dir="parsed_data",
    output_dir="heatmap_256x256_maxproj",
    resolution=256,
    aggregate_frames=True,
    frames_per_sample=5,
    aggregation_mode="max",
    bin_statistic="max"
)
```

## Input Data Format

### Point Cloud CSV (`*_pointcloud.csv`)

Required columns:
- `frame_num` (int): Frame number
- `x` (float): X coordinate (meters)
- `z` (float): Z coordinate (meters)
- `velocity` (float): Doppler velocity (m/s)
- `snr` (float): Signal-to-noise ratio (dB)

### Summary CSV (`*_summary.csv`)

Required columns:
- `frame_num` (int): Frame number
- `ground_truth` (str/int): Label (e.g., "3_person" or 3)

## Output Structure

```
heatmap_dataset/
├── groundtruth.csv           # Metadata with labels and frame info
├── radar_f0001_0person.png   # Heatmap images
├── radar_f0002_1person.png
├── radar_f0003_2person.png
└── ...
```

### Ground Truth CSV Columns

- `filename`: Output filename (without extension)
- `frame_num` or `frames`: Frame number(s) used
- `label`: Person count (0, 1, 2, 3, ...)
- `split`: Data split (train/val/test/all)
- `base_name`: Original file identifier
- `aggregation_mode`: Mode used (if aggregation enabled)

## Workflow Integration

This tool is designed to work in a pipeline:

1. **Data Parsing** → Parse raw radar data to CSV format
2. **Dataset Generation** → This tool (heatmap images + labels)
3. **Model Training** → Use generated images for CNN/deep learning
4. **Evaluation** → Test on generated validation/test images

## Technical Details

### Heatmap Generation Process

1. **Point Cloud Loading**: Read CSV files with radar detections
2. **Spatial Filtering**: Apply bounding box and velocity filters
3. **2D Binning**: Use `scipy.stats.binned_statistic_2d` with configurable statistics
4. **Frame Aggregation** (if enabled): Combine multiple frames based on mode
5. **Image Rendering**: Generate clean axis-free heatmap images
6. **Metadata Export**: Create CSV with ground truth labels

### Color Mapping

Custom blue-to-yellow gradient colormap:
- **Blue**: Low intensity (background)
- **Cyan**: Medium-low intensity
- **Yellow**: High intensity (strong detections)

### Coordinate System

- **X-axis**: Horizontal (left-right), typically -6m to +6m
- **Z-axis**: Depth (forward-backward), typically -6m to +6m
- **Origin**: Radar sensor position (0, 0)

## Output Example
<img width="472" height="472" alt="image" src="https://github.com/user-attachments/assets/94c861b1-6bb6-4aaa-8b78-1376aacffa89" />
