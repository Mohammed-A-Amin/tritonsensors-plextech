#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional

# =========================
# CONFIGURATION (edit here)
# =========================

# ========== INPUT/OUTPUT PATHS ==========
INPUT_DIR = "split_data"  # Directory containing train/val/test CSV files from parser
OUTPUT_DIR = "processed_dataset"  # Directory to save processed .npy files

# ========== DATA PROCESSING OPTIONS ==========
# Which CSV file type to process (from combined_parser_splitter output)
USE_POINTCLOUD = True  # Process _pointcloud.csv files
USE_TRACKS = False     # Process _tracks.csv files (not recommended for ML)

# ========== FRAME STACKING CONFIGURATION ==========
FRAMES_TO_STACK = 8  # Number of consecutive frames to stack together

# ========== ZERO-PADDING CONFIGURATION ==========
# Fixed number of points per stacked sample (after stacking frames)
POINTS_PER_SAMPLE = 1024  # Target size after stacking and padding
PADDING_VALUE = 0.0    # Value to use for padding

# ========== FEATURE SELECTION ==========
# Select which features to include in the dataset
FEATURES = ['x', 'y', 'z', 'velocity']  # Available: x, y, z, velocity, snr, noise

# ========== LABEL CONFIGURATION ==========
# Label column name in the summary CSV files
LABEL_COLUMN = 'ground_truth'  # Column name in *_summary_all.csv
LABEL_TYPE = 'count'  # 'count' for people counting, 'class' for classification
LABEL_AGGREGATION = 'last'  # 'last', 'max', 'mean', or 'first' for stacked frames

# ========== OUTPUT OPTIONS ==========
SAVE_SEPARATE_SPLITS = True   # Save train/val/test as separate .npy files
SAVE_COMBINED = False          # Save all data with split column
INCLUDE_FRAME_INFO = True      # Save frame numbers and file names for reference

# =========================
# HELPER FUNCTIONS
# =========================

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def pad_or_truncate_points(points_data: np.ndarray, target_size: int, 
                          padding_value: float = 0.0) -> np.ndarray:
    """
    Pad or truncate point cloud data to fixed size.

    Args:
        points_data: Array of shape (num_points, num_features)
        target_size: Target number of points
        padding_value: Value to use for padding

    Returns:
        Array of shape (target_size, num_features)
    """
    current_size, num_features = points_data.shape

    if current_size < target_size:
        # Pad with padding_value
        padding = np.full((target_size - current_size, num_features), 
                         padding_value, dtype=points_data.dtype)
        padded = np.vstack([points_data, padding])
        return padded

    elif current_size > target_size:
        # Truncate (keep first N points)
        # Alternative: randomly sample points or use farthest point sampling
        return points_data[:target_size]

    return points_data

def aggregate_labels(labels: np.ndarray, method: str = 'last') -> int:
    """
    Aggregate labels from multiple frames into a single label.

    Args:
        labels: Array of labels from consecutive frames
        method: Aggregation method ('last', 'max', 'mean', 'first')

    Returns:
        Single aggregated label
    """
    if method == 'last':
        return labels[-1]
    elif method == 'first':
        return labels[0]
    elif method == 'max':
        return np.max(labels)
    elif method == 'mean':
        return int(np.round(np.mean(labels)))
    else:
        return labels[-1]  # Default to last

def get_summary_file_path(csv_path: str) -> str:
    """
    Get the corresponding summary file path for a given CSV file.

    Args:
        csv_path: Path to the pointcloud CSV file (e.g., "file_train_pointcloud.csv")

    Returns:
        Path to the summary file (e.g., "file_summary_all.csv")
    """
    directory = os.path.dirname(csv_path)
    filename = os.path.basename(csv_path)

    # Remove split suffix and file type suffix
    base_name = filename.replace('_pointcloud_train.csv', '').replace('_pointcloud_val.csv', '').replace('_pointcloud_test.csv', '').replace('_tracks_train.csv', '').replace('_tracks_val.csv', '').replace('_tracks_test.csv', '').replace('_pointcloud.csv', '').replace('_tracks.csv', '')

    summary_filename = f"{base_name}_summary_all.csv"
    summary_path = os.path.join(directory, summary_filename)

    return summary_path

def load_ground_truth(summary_path: str) -> Optional[pd.DataFrame]:
    """
    Load ground truth data from summary file.

    Args:
        summary_path: Path to the summary CSV file

    Returns:
        DataFrame with frame_num and ground_truth columns, or None if not found
    """
    if not os.path.exists(summary_path):
        print(f"  ⚠ Summary file not found: {os.path.basename(summary_path)}")
        return None

    try:
        df = pd.read_csv(summary_path)

        if 'frame_num' not in df.columns or LABEL_COLUMN not in df.columns:
            print(f"  ⚠ Required columns missing in summary file")
            return None

        return df[['frame_num', LABEL_COLUMN]]

    except Exception as e:
        print(f"  ⚠ Error loading summary file: {e}")
        return None

def process_pointcloud_csv(csv_path: str, features: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single pointcloud CSV file with frame stacking.

    Args:
        csv_path: Path to CSV file
        features: List of feature names to extract

    Returns:
        Tuple of (processed_data, labels, frame_nums)
        - processed_data: (num_samples, points_per_sample, num_features)
        - labels: (num_samples,) or None
        - frame_nums: (num_samples, frames_to_stack) - start and end frame numbers
    """
    print(f"  Loading: {os.path.basename(csv_path)}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ⚠ Error loading file: {e}")
        return None, None, None

    if df.empty:
        print(f"  ⚠ Empty dataframe")
        return None, None, None

    # Check if required columns exist
    missing_cols = [f for f in features if f not in df.columns]
    if missing_cols:
        print(f"  ⚠ Missing columns: {missing_cols}")
        return None, None, None

    if 'frame_num' not in df.columns:
        print(f"  ⚠ No 'frame_num' column found")
        return None, None, None

    # Load ground truth from summary file
    summary_path = get_summary_file_path(csv_path)
    ground_truth_df = load_ground_truth(summary_path)

    # Get unique frames (sorted)
    unique_frames = sorted(df['frame_num'].unique())
    num_frames = len(unique_frames)
    num_features = len(features)

    # Calculate number of stacked samples
    num_samples = num_frames - FRAMES_TO_STACK + 1

    if num_samples <= 0:
        print(f"  ⚠ Not enough frames ({num_frames}) for stacking {FRAMES_TO_STACK} frames")
        return None, None, None

    # Initialize arrays
    processed_samples = np.zeros((num_samples, POINTS_PER_SAMPLE, num_features), dtype=np.float32)
    frame_ranges = np.zeros((num_samples, 2), dtype=np.int32)  # [start_frame, end_frame]

    # Extract labels if available
    labels = None
    if ground_truth_df is not None:
        labels = np.zeros(num_samples, dtype=np.int32)

    # Process each stacked sample
    for sample_idx in range(num_samples):
        # Get frames for this sample
        frame_indices = range(sample_idx, sample_idx + FRAMES_TO_STACK)
        frames_to_process = [unique_frames[i] for i in frame_indices]

        # Store frame range
        frame_ranges[sample_idx] = [frames_to_process[0], frames_to_process[-1]]

        # Collect all points from stacked frames
        stacked_points = []
        sample_labels = []

        for frame_num in frames_to_process:
            frame_data = df[df['frame_num'] == frame_num].copy()

            # Extract features
            feature_arrays = []
            for feat in features:
                values = frame_data[feat].values
                feature_arrays.append(values)

            # Stack features (num_points_in_frame, num_features)
            if len(feature_arrays[0]) > 0:  # Check if frame has points
                frame_features = np.column_stack(feature_arrays).astype(np.float32)
                stacked_points.append(frame_features)

            # Collect label for this frame
            if labels is not None and ground_truth_df is not None:
                gt_row = ground_truth_df[ground_truth_df['frame_num'] == frame_num]
                if not gt_row.empty:
                    sample_labels.append(gt_row[LABEL_COLUMN].iloc[0])
                else:
                    sample_labels.append(-1)

        # Concatenate all points from stacked frames
        if stacked_points:
            combined_points = np.vstack(stacked_points)
        else:
            # If no points in any frame, create empty array
            combined_points = np.zeros((0, num_features), dtype=np.float32)

        # Pad or truncate to target size
        combined_points = pad_or_truncate_points(combined_points, POINTS_PER_SAMPLE, PADDING_VALUE)

        # Store
        processed_samples[sample_idx] = combined_points

        # Aggregate labels from stacked frames
        if labels is not None and sample_labels:
            labels[sample_idx] = aggregate_labels(np.array(sample_labels), LABEL_AGGREGATION)

    print(f"  ✓ Processed {num_samples} samples (stacked from {num_frames} frames)")
    print(f"    Each sample contains {FRAMES_TO_STACK} consecutive frames with {POINTS_PER_SAMPLE} points")

    return processed_samples, labels, frame_ranges

def collect_files_by_split(input_dir: str, pattern: str) -> Dict[str, List[str]]:
    """
    Collect CSV files organized by split (train/val/test).

    Args:
        input_dir: Directory containing CSV files
        pattern: File pattern to match (e.g., '_pointcloud')

    Returns:
        Dictionary mapping split names to list of file paths
    """
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return {}

    files_by_split = {'train': [], 'val': [], 'test': []}

    all_files = os.listdir(input_dir)

    for filename in all_files:
        if pattern in filename and filename.endswith('.csv'):
            filepath = os.path.join(input_dir, filename)

            # Determine split based on filename
            if '_train' in filename and '_pointcloud' in filename:
                files_by_split['train'].append(filepath)
            elif '_val' in filename and '_pointcloud' in filename:
                files_by_split['val'].append(filepath)
            elif '_test' in filename and '_pointcloud' in filename:
                files_by_split['test'].append(filepath)

    return files_by_split

def process_split_data(file_list: List[str], features: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Process all files for a given split.

    Args:
        file_list: List of CSV file paths
        features: List of feature names

    Returns:
        Tuple of (combined_data, combined_labels)
    """
    all_data = []
    all_labels = []

    for filepath in file_list:
        data, labels, frame_ranges = process_pointcloud_csv(filepath, features)

        if data is not None:
            all_data.append(data)
            if labels is not None:
                all_labels.append(labels)

    if not all_data:
        return None, None

    # Concatenate all data
    combined_data = np.concatenate(all_data, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0) if all_labels else None

    return combined_data, combined_labels

# =========================
# MAIN FUNCTION
# =========================

def main():
    """Main execution function."""
    print(f"\n{'='*80}")
    print(f"RADAR DATASET GENERATOR WITH FRAME STACKING")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Input directory: {INPUT_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Frames to stack: {FRAMES_TO_STACK}")
    print(f"  Points per sample (after stacking): {POINTS_PER_SAMPLE}")
    print(f"  Features: {FEATURES}")
    print(f"  Label aggregation: {LABEL_AGGREGATION}")
    print(f"  Ground truth source: *_summary_all.csv files")
    print(f"{'='*80}\n")

    ensure_dir(OUTPUT_DIR)

    # Determine which files to process
    file_pattern = '_pointcloud' if USE_POINTCLOUD else '_tracks'

    # Collect files by split
    print("Collecting input files...")
    files_by_split = collect_files_by_split(INPUT_DIR, file_pattern)

    for split, files in files_by_split.items():
        if files:
            print(f"  {split}: {len(files)} file(s)")
    print()

    # Process each split
    for split_name in ['train', 'val', 'test']:
        file_list = files_by_split.get(split_name, [])

        if not file_list:
            print(f"No {split_name} files found, skipping...\n")
            continue

        print(f"{'='*80}")
        print(f"Processing {split_name.upper()} split...")
        print(f"{'='*80}")

        # Process all files in this split
        data, labels = process_split_data(file_list, FEATURES)

        if data is None:
            print(f"⚠ No valid data processed for {split_name} split\n")
            continue

        print(f"\n{split_name.upper()} split statistics:")
        print(f"  Data shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Memory size: {data.nbytes / 1024 / 1024:.2f} MB")

        if labels is not None:
            print(f"  Labels shape: {labels.shape}")
            unique_labels = np.unique(labels)
            print(f"  Unique labels: {unique_labels}")
            if -1 in unique_labels:
                missing_count = np.sum(labels == -1)
                print(f"  ⚠ Missing ground truth: {missing_count} samples")

        # Save processed data
        if SAVE_SEPARATE_SPLITS:
            data_path = os.path.join(OUTPUT_DIR, f'data_{split_name}.npy')
            np.save(data_path, data)
            print(f"  ✓ Saved: {data_path}")

            if labels is not None:
                labels_path = os.path.join(OUTPUT_DIR, f'labels_{split_name}.npy')
                np.save(labels_path, labels)
                print(f"  ✓ Saved: {labels_path}")

        print()

    # Save configuration for reference
    config_path = os.path.join(OUTPUT_DIR, 'dataset_config.txt')
    with open(config_path, 'w') as f:
        f.write(f"Dataset Configuration\n")
        f.write(f"{'='*80}\n")
        f.write(f"Frames stacked per sample: {FRAMES_TO_STACK}\n")
        f.write(f"Points per sample (after stacking and padding): {POINTS_PER_SAMPLE}\n")
        f.write(f"Features: {FEATURES}\n")
        f.write(f"Number of features: {len(FEATURES)}\n")
        f.write(f"Label aggregation method: {LABEL_AGGREGATION}\n")
        f.write(f"Ground truth source: *_summary_all.csv files\n")
        f.write(f"Data shape per sample: ({POINTS_PER_SAMPLE}, {len(FEATURES)})\n")

    print(f"{'='*80}")
    print(f"✓ DATASET GENERATION COMPLETE!")
    print(f"  Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Configuration saved: {config_path}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
