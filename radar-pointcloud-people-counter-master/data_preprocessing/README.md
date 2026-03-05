# Radar Data Preprocessing Pipeline

This module converts raw JSON radar tracking data into CSV format with train/val/test splitting. The preprocessed CSV files serve as input for model-specific dataset generators.

## Pipeline Overview

The preprocessing pipeline performs two operations:

1. **Parse JSON to CSV**: Convert TI mmWave radar JSON output to structured CSV files
2. **Split Data**: Divide datasets into train/validation/test sets by frame sequence

## Features

- Parse TI mmWave radar JSON output to CSV format
- Extract point cloud, track, and summary data
- Sequential train/val/test splitting by frame numbers (maintains temporal consistency)
- Configurable frame filtering and radar parameters
- Batch processing with ground truth labels
- Floor elevation correction for mounted radars

## Requirements

```bash
pip install pandas numpy
```

## Usage

### Running the Preprocessor

Edit the configuration section at the top of `combined_parser_splitter.py`:

1. Set input JSON files and ground truth labels in `INPUT_FILES_AND_LABELS`
2. Configure split ratios (default: 70% train, 0% val, 30% test)
3. Adjust radar parameters (height, vertical axis)
4. Choose which outputs to generate (pointcloud, tracks, summary)

Run the script:

```bash
python combined_parser_splitter.py
```

### Configuration Options

**Operation Mode:**
- `RUN_PARSER`: Convert JSON to CSV
- `RUN_SPLITTER`: Split CSV into train/val/test sets

**Input/Output:**
- `INPUT_FILES_AND_LABELS`: List of (filepath, label) tuples
  ```python
  INPUT_FILES_AND_LABELS = [
      (r"ParsaOffice_TwoPeople_T0.json", "2_person"),
      (r"ParsaOffice_EmptyRoom_T0.json", "0_person"),
  ]
  ```
- `PARSED_OUTPUT_DIR`: Output directory for parsed CSVs (default: `parsed_data`)
- `SPLIT_OUTPUT_DIR`: Output directory for split datasets (default: `split_data`)

**Frame Filtering:**
- `SKIP_FIRST`: Number of frames to skip at the beginning
- `SKIP_LAST`: Number of frames to skip at the end

**Radar Configuration:**
- `RADAR_HEIGHT_M`: Sensor mounting height in meters (default: 2.7)
- `VERTICAL_AXIS`: Which axis represents height (default: "y" for TI 6843)
- `APPLY_FLOOR_ELEVATION`: Convert coordinates to floor reference (default: True)

**Data Splitting:**
- `TRAIN_SPLIT_PERCENT`: Training set percentage (default: 0.7)
- `VAL_SPLIT_PERCENT`: Validation set percentage (default: 0.0)
- `TEST_SPLIT_PERCENT`: Test set percentage (default: 0.3)

**Output Options:**
- `CREATE_POINTCLOUD_CSV`: Extract raw point clouds (recommended for ML)
- `CREATE_TRACK_CSV`: Extract tracked objects
- `CREATE_SUMMARY_CSV`: Frame-level statistics with ground truth
- `CREATE_SEPARATE_FILES`: Generate `_train.csv`, `_val.csv`, `_test.csv` files
- `CREATE_COMBINED_FILE`: Generate `_all.csv` with split column

## Output Format

### Point Cloud CSV
- **Columns**: `frame_num`, `x`, `y`, `z`, `velocity`, `snr`
- **Use case**: Primary input for ML models
- **Example**: `filename_pointcloud_train.csv`

### Track CSV
- **Columns**: `frame_num`, `track_id`, `x`, `y`, `z`, `vel_x`, `vel_y`, `vel_z`, `acc_x`, `acc_y`, `acc_z`
- **Use case**: Analyzed tracked objects with velocity/acceleration
- **Example**: `filename_tracks_train.csv`

### Summary CSV
- **Columns**: `frame_num`, `num_points`, `num_tracked_objects`, `ground_truth`
- **Use case**: Frame-level statistics and ground truth labels
- **Example**: `filename_summary_all.csv`

### Split Files
- **Separate files**: `*_train.csv`, `*_val.csv`, `*_test.csv`
- **Combined file**: `*_all.csv` (includes `split` column with values: train/val/test)

## Data Flow

```
Raw JSON Files
      ↓
combined_parser_splitter.py
      ↓
CSV Files (parsed_data/)
      ↓
CSV Files (split_data/) ← train/val/test splits
      ↓
Model-Specific Dataset Generators
      ↓
NumPy Arrays for Training
```

## Example Workflow

```python
# Step 1: Configure combined_parser_splitter.py
INPUT_FILES_AND_LABELS = [
    (r"data/ParsaOffice_TwoPeople_T0.json", "2_person"),
    (r"data/ParsaOffice_SinglePerson_T0.json", "1_person"),
    (r"data/ParsaOffice_EmptyRoom_T0.json", "0_person"),
]

TRAIN_SPLIT_PERCENT = 0.7
VAL_SPLIT_PERCENT = 0.15
TEST_SPLIT_PERCENT = 0.15

RADAR_HEIGHT_M = 2.7
VERTICAL_AXIS = "y"
APPLY_FLOOR_ELEVATION = True

# Step 2: Run preprocessing
# python combined_parser_splitter.py

# Output structure:
# split_data/
#   ├── ParsaOffice_TwoPeople_T0_pointcloud_train.csv
#   ├── ParsaOffice_TwoPeople_T0_pointcloud_test.csv
#   ├── ParsaOffice_TwoPeople_T0_summary_all.csv
#   └── ... (other files)
```

## Important Notes

**Ground Truth Labels:**
- Ground truth is parsed from label strings in format `"N_person"` where N is the count
- Labels are stored in summary CSV files for each recording
- All frames from the same recording share the same ground truth value

**Temporal Consistency:**
- Splitting is **sequential** by frame number (not random)
- This maintains temporal order, which is important for time-series models
- Train/val/test sets contain consecutive frame sequences

**Coordinate System:**
- Floor elevation correction converts radar coordinates to floor-referenced system
- Vertical coordinate = `RADAR_HEIGHT_M - measured_height`
- Ensures consistent spatial reference across different mounting heights

**Memory Efficiency:**
- Batch processing avoids loading all JSON data into memory at once
- CSV format allows selective column loading in downstream processing

## Next Steps

After preprocessing, use model-specific dataset generators to create ML-ready arrays:

- **For CNN-LSTM models**: Use `cnn_lstm/cnn_lstm_dataset_generator.py` (with frame stacking)
- **For other architectures**: Create custom dataset generators as needed

Each model folder contains its own dataset generator tailored to the model's input requirements.
