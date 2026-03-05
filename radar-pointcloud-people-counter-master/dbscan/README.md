# DBSCAN People Counter

Density-based spatial clustering approach for counting people from radar point clouds. This method combines multi-frame data and uses automatic parameter tuning to achieve robust detection across various scenarios.

## Overview

This implementation performs people counting by:
1. **Loading** multiple radar JSON files with ground truth labels
2. **Filtering** point clouds by spatial boundaries and velocity thresholds
3. **Clustering** combined frames using DBSCAN algorithm
4. **Optimizing** parameters automatically or via grid search
5. **Evaluating** performance with comprehensive metrics

## Key Features

### Filtering Capabilities
- **Spatial Bounding**: Filter points by X, Y, Z coordinate limits
- **Velocity Filtering**: Remove static clutter based on velocity threshold
- **Frame Edge Filtering**: Skip initial/final frames if needed

### Parameter Tuning
- **Fixed Mode**: Use pre-tuned parameters for inference
- **Auto Search**: Automatically estimate optimal epsilon using k-distance
- **Grid Search**: Exhaustive search over parameter ranges

### Feature Selection
- **2D Clustering**: Use (X, Z) for horizontal plane analysis
- **3D Clustering**: Use (X, Y, Z) for full spatial analysis
- **Custom Features**: Select any combination of available features

### Optimization Metrics
- **GFR**: Good Frame Rate (exact match percentage)
- **GFR±1**: Tolerance within 1 person
- **GFR±2**: Tolerance within 2 people
- **F1-macro**: Macro-averaged F1 score
- **F1-weighted**: Weighted F1 score

### Output Reports
- Per-frame predictions with cluster details
- Per-file performance metrics
- Per-class (ground truth) grouped statistics
- Confusion matrices and classification reports
- Detailed JSON reports with cluster centroids and sizes

## Configuration

### Basic Setup

**Input Files** (Update these paths to your own data):
```
INPUT_FILES_AND_LABELS = [
(r"/path/to/your/radar_data_3people.json", "3_person"),
(r"/path/to/your/radar_data_1person.json", "1_person"),
# Add your files here...
]
```

*Note: The paths shown in the code are sample inputs only. Replace them with your actual data paths.*

**Operation Mode**:
```
USE_FIXED_PARAMS = False # False for tuning, True for inference
```

**Fixed Parameters** (when USE_FIXED_PARAMS = True):
```
FIXED_EPS = 0.3 # DBSCAN epsilon parameter
FIXED_MIN_SAMPLES = 5 # Minimum samples per cluster
```

### Spatial Filtering
```
ENABLE_SPATIAL_FILTER = True
X_LIM = [-1.5, 1.5] # Meters, lateral range
Y_LIM = [0.0, 2.7] # Meters, height range (floor-referenced)
Z_LIM = [-3.0, 3.0] # Meters, depth range
```

### Velocity Filtering
```
ENABLE_VELOCITY_FILTER = True
VELOCITY_THRESHOLD = 0.01 # m/s, minimum velocity to keep point
```

### Parameter Search Configuration

**Auto Mode**:
```
SEARCH_MODE = "auto"
AUTO_MIN_SAMPLES_CANDIDATES =​ [0, 3, 4, 5, 6, 8]
AUTO_TOPK_EPS_AGG = "p50" # p50, p75, median, mean
AUTO_SCORE_METRIC = "f1_macro" # gfr, gfr1, gfr2, f1_macro, f1_weighted
```

**Grid Mode**:
```
SEARCH_MODE = "grid"
GRID_EPS_LIST = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
GRID_MIN_SAMPLES_LIST =​ [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
GRID_SCORE_METRIC = "gfr"
```

### Feature Selection
```
USE_FEATURES = ("x", "y", "z") # 3D clustering

USE_FEATURES = ("x", "z") # 2D clustering (horizontal plane only)
```

### DBSCAN Parameters
```
NUM_FRAMES_FOR_DBSCAN = 4 # Frames to combine per clustering
CLUSTER_MIN_SIZE_FOR_COUNT = 1 # Min points to count as person
```

## Usage Examples

### Example 1: Parameter Tuning with Auto Search
```
Configuration
USE_FIXED_PARAMS = False
SEARCH_MODE = "auto"
AUTO_SCORE_METRIC = "gfr" # Optimize for GFR
USE_FEATURES = ("x", "y", "z")

Run
python dbscan_people_counter.py
```

**Output**: Best parameters will be printed and saved to `outputs/search_report.json`

### Example 2: Inference with Fixed Parameters
```
Configuration
USE_FIXED_PARAMS = True
FIXED_EPS = 0.3
FIXED_MIN_SAMPLES = 5

Run
python dbscan_people_counter.py
```

### Example 3: Grid Search with Custom Range
```
Configuration
USE_FIXED_PARAMS = False
SEARCH_MODE = "grid"
GRID_EPS_LIST = [0.2, 0.25, 0.3, 0.35, 0.4]
GRID_MIN_SAMPLES_LIST =​ [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
GRID_SCORE_METRIC = "gfr" # Optimize for GFR

Run
python dbscan_people_counter.py
```


## Outputs

All results are saved to the `outputs/` directory:

| File | Description |
|------|-------------|
| `combined_frame_summary.csv` | Frame-by-frame predictions with cluster counts |
| `search_report.json` | Parameter search results and best configuration |
| `per_file_metrics.csv` | Performance metrics for each input file |
| `per_gt_metrics.csv` | Metrics grouped by ground truth people count |
| `combined_metrics.json` | Overall performance statistics |
| `detailed_results.json` | Complete results with cluster centroids and sizes |
| `per_class_metrics.csv` | Precision, recall, F1 per people count class |

## Performance Tips

1. **Start with Auto Search**: Use auto mode to quickly find good parameters
2. **3D vs 2D**: Try both feature sets - 2D often works well for horizontal tracking
3. **Velocity Threshold**: Adjust based on your scene (higher for static environments)
4. **Frame Fusion**: Increase `NUM_FRAMES_FOR_DBSCAN` for more stable detection (slower)
5. **Spatial Bounds**: Tight bounds improve accuracy by excluding walls/clutter

## Algorithm Details

### Multi-Frame Fusion
Combines N consecutive frames before clustering to:
- Increase point density for sparse radar data
- Improve cluster stability across temporal variations
- Reduce impact of temporary occlusions

### Epsilon Estimation
In auto mode, epsilon is estimated using k-distance graphs:
1. Compute k-nearest neighbor distances for all points
2. Sort distances and identify "elbow" in curve using second derivative
3. Aggregate estimates across frames (median, p50, p75, etc.)

### Coordinate System
For ceiling-mounted radars:
- Y-axis represents height (radar to floor)
- Automatically converts to floor-referenced elevation
- Configurable via `MOUNTING`, `VERTICAL_AXIS`, `RADAR_HEIGHT_M`

## Troubleshooting

**Issue**: Low accuracy with default parameters
- **Solution**: Run parameter search with auto or grid mode

**Issue**: Too many false detections
- **Solution**: Increase `VELOCITY_THRESHOLD` or tighten spatial bounds

**Issue**: Missing detections
- **Solution**: Decrease `CLUSTER_MIN_SIZE_FOR_COUNT` or increase `NUM_FRAMES_FOR_DBSCAN`

**Issue**: Slow performance
- **Solution**: Use fixed parameters mode, reduce grid search ranges

## Requirements
```
numpy
pandas
scikit-learn
```
install with:
```
pip install numpy pandas scikit-learn
```