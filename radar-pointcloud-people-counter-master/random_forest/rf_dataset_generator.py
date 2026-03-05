#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

# =========================
# CONFIGURATION (edit here)
# =========================

# ========== INPUT/OUTPUT PATHS ==========
INPUT_DIR = "split_data"  # Directory containing _pointcloud_train.csv, etc.
OUTPUT_DIR = "classical_ml_dataset"  # Directory to save feature CSVs

# ========== PREPROCESSING FILTERS ==========
# Spatial filtering: Remove points outside room boundaries
ENABLE_SPATIAL_FILTER = False
SPATIAL_LIMITS = {
    'x_min': -3.0,  # meters
    'x_max': 3.0,
    'y_min': 0.0,
    'y_max': 6.0,
    'z_min': -1.0,
    'z_max': 2.5
}

# Velocity filtering: Remove static/low-velocity points (denoising)
ENABLE_VELOCITY_FILTER = False
VELOCITY_THRESHOLD = 0.1  # m/s - points with |velocity| < threshold are removed

# SNR filtering: Remove low-SNR points (optional additional denoising)
ENABLE_SNR_FILTER = False
SNR_THRESHOLD = 10.0  # dB - points with SNR < threshold are removed

# ========== FEATURE COMPUTATION ==========
# Add num_tracked_objects from summary as a feature
ADD_TRACKED_OBJECTS_FEATURE = True  # If True, adds num_tracked_objects as feature

# Add filtering statistics as features
ADD_FILTER_STATS = False  # If True, adds counts before/after filtering

# Epsilon for division to avoid zero-division
EPS = 1e-8

# Include metadata columns (frame_num, source_file) in output CSV
INCLUDE_METADATA = False

# =========================
# HELPER FUNCTIONS
# =========================

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def compute_range(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Compute range (distance from origin) for points."""
    return np.sqrt(x**2 + y**2 + z**2)

def apply_spatial_filter(frame_df: pd.DataFrame, limits: Dict[str, float]) -> pd.DataFrame:
    """
    Filter points based on spatial boundaries (room limits).
    
    Args:
        frame_df: DataFrame with point cloud data
        limits: Dictionary with x_min, x_max, y_min, y_max, z_min, z_max
    
    Returns:
        Filtered DataFrame
    """
    if not ENABLE_SPATIAL_FILTER:
        return frame_df
    
    mask = (
        (frame_df['x'] >= limits['x_min']) & (frame_df['x'] <= limits['x_max']) &
        (frame_df['y'] >= limits['y_min']) & (frame_df['y'] <= limits['y_max']) &
        (frame_df['z'] >= limits['z_min']) & (frame_df['z'] <= limits['z_max'])
    )
    
    return frame_df[mask].copy()

def apply_velocity_filter(frame_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Filter points based on velocity magnitude (remove static points).
    
    Args:
        frame_df: DataFrame with point cloud data
        threshold: Minimum absolute velocity (m/s)
    
    Returns:
        Filtered DataFrame
    """
    if not ENABLE_VELOCITY_FILTER:
        return frame_df
    
    mask = np.abs(frame_df['velocity']) >= threshold
    return frame_df[mask].copy()

def apply_snr_filter(frame_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Filter points based on SNR (remove low-quality detections).
    
    Args:
        frame_df: DataFrame with point cloud data
        threshold: Minimum SNR (dB)
    
    Returns:
        Filtered DataFrame
    """
    if not ENABLE_SNR_FILTER:
        return frame_df
    
    mask = frame_df['snr'] >= threshold
    return frame_df[mask].copy()

def preprocess_frame(frame_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Apply all enabled preprocessing filters to frame data.
    
    Returns:
        Filtered DataFrame and original point count
    """
    original_count = len(frame_df)
    
    # Apply filters in sequence
    frame_df = apply_spatial_filter(frame_df, SPATIAL_LIMITS)
    frame_df = apply_velocity_filter(frame_df, VELOCITY_THRESHOLD)
    frame_df = apply_snr_filter(frame_df, SNR_THRESHOLD)
    
    return frame_df, original_count

def extract_features_per_frame(frame_df: pd.DataFrame, original_count: int = 0) -> Dict[str, float]:
    """
    Extract features for a single frame's points (all detected points after filtering).
    
    Args:
        frame_df: DataFrame with columns ['x', 'y', 'z', 'velocity', 'snr', 'noise', 'track_idx']
        original_count: Original number of points before filtering
    
    Returns:
        Dict of feature names to values
    """
    # Initialize features with filtering stats
    features = {}
    
    if ADD_FILTER_STATS:
        features['num_points_before_filter'] = float(original_count)
        features['num_points_after_filter'] = float(len(frame_df))
        features['filter_rejection_ratio'] = (original_count - len(frame_df)) / (original_count + EPS)
    
    if frame_df.empty:
        # Default values for empty frames
        default_features = {
            'num_total_points': 0.0,
            'mean_range': 0.0, 'std_range': 0.0,
            'mean_doppler': 0.0, 'std_doppler': 0.0,
            'mean_snr': 0.0, 'std_snr': 0.0,
            'mean_x': 0.0, 'std_x': 0.0, 'min_x': 0.0, 'max_x': 0.0,
            'mean_y': 0.0, 'std_y': 0.0, 'min_y': 0.0, 'max_y': 0.0,
            'mean_z': 0.0, 'std_z': 0.0, 'min_z': 0.0, 'max_z': 0.0,
            'ratio_x_y': 0.0, 'ratio_x_z': 0.0, 'ratio_z_y': 0.0,
            'magnitude_range': 0.0,
            'magnitude_doppler': 0.0,
            'magnitude_snr': 0.0,
            'magnitude_x': 0.0, 'magnitude_y': 0.0, 'magnitude_z': 0.0,
        }
        features.update(default_features)
        return features
    
    # Extract all points
    x = frame_df['x'].values
    y = frame_df['y'].values
    z = frame_df['z'].values
    velocity = frame_df['velocity'].values  # Doppler
    snr = frame_df['snr'].values
    noise = frame_df['noise'].values
    
    num_total = len(frame_df)
    
    # Compute range
    range_vals = compute_range(x, y, z)
    
    # Helper functions for safe statistics
    def safe_std(vals):
        return np.std(vals) if len(vals) > 1 else 0.0
    
    def safe_mean(vals):
        return np.mean(vals) if len(vals) > 0 else 0.0
    
    def safe_min_max(vals):
        return (np.min(vals) if len(vals) > 0 else 0.0, 
                np.max(vals) if len(vals) > 0 else 0.0)
    
    # Feature 1: Number of total points detected (after filtering)
    features['num_total_points'] = float(num_total)
    
    # Features 3, 6: Mean and STD of range
    features['mean_range'] = safe_mean(range_vals)
    features['std_range'] = safe_std(range_vals)
    
    # Features 4, 7: Mean and STD of Doppler (velocity)
    features['mean_doppler'] = safe_mean(velocity)
    features['std_doppler'] = safe_std(velocity)
    
    # Features 5, 8: Mean and STD of SNR
    features['mean_snr'] = safe_mean(snr)
    features['std_snr'] = safe_std(snr)
    
    # Features 9, 12, 15, 18: Mean, STD, Min, Max of X
    features['mean_x'] = safe_mean(x)
    features['std_x'] = safe_std(x)
    features['min_x'], features['max_x'] = safe_min_max(x)
    
    # Features 10, 13, 16, 19: Mean, STD, Min, Max of Y
    features['mean_y'] = safe_mean(y)
    features['std_y'] = safe_std(y)
    features['min_y'], features['max_y'] = safe_min_max(y)
    
    # Features 11, 14, 17, 20: Mean, STD, Min, Max of Z
    features['mean_z'] = safe_mean(z)
    features['std_z'] = safe_std(z)
    features['min_z'], features['max_z'] = safe_min_max(z)
    
    # Features 21, 22, 23: Ratio between X and Y, X and Z, Z and Y
    width_x = features['max_x'] - features['min_x']
    width_y = features['max_y'] - features['min_y']
    width_z = features['max_z'] - features['min_z']
    
    features['ratio_x_y'] = width_x / (width_y + EPS)
    features['ratio_x_z'] = width_x / (width_z + EPS)
    features['ratio_z_y'] = width_z / (width_y + EPS)
    
    # Features 24, 25, 26, 27, 28, 29: Magnitudes
    features['magnitude_range'] = safe_mean(np.abs(range_vals))
    features['magnitude_doppler'] = safe_mean(np.abs(velocity))
    features['magnitude_snr'] = safe_mean(np.abs(snr))
    features['magnitude_x'] = safe_mean(np.abs(x))
    features['magnitude_y'] = safe_mean(np.abs(y))
    features['magnitude_z'] = safe_mean(np.abs(z))
    
    return features

def get_summary_file_path(csv_path: str) -> str:
    """
    Get the corresponding summary file path.
    Assumes summary_all.csv exists for labels.
    """
    directory = os.path.dirname(csv_path)
    filename = os.path.basename(csv_path)
    
    # Replace split and type suffixes
    base_name = filename.replace('_pointcloud_train.csv', '').replace('_pointcloud_val.csv', '') \
                        .replace('_pointcloud_test.csv', '').replace('_pointcloud.csv', '')
    
    summary_filename = f"{base_name}_summary_all.csv"
    return os.path.join(directory, summary_filename)

def load_ground_truth(summary_path: str) -> Optional[pd.DataFrame]:
    """
    Load ground truth and optional num_tracked_objects from summary CSV.
    """
    if not os.path.exists(summary_path):
        print(f" ⚠ Summary file not found: {os.path.basename(summary_path)}")
        return None
    
    try:
        df = pd.read_csv(summary_path)
        required_cols = ['frame_num', 'ground_truth']
        
        if not all(col in df.columns for col in required_cols):
            print(f" ⚠ Required columns missing in summary file")
            return None
        
        # Include num_tracked_objects if available and requested
        cols_to_return = ['frame_num', 'ground_truth']
        if ADD_TRACKED_OBJECTS_FEATURE and 'num_tracked_objects' in df.columns:
            cols_to_return.append('num_tracked_objects')
        
        return df[cols_to_return]
    except Exception as e:
        print(f" ⚠ Error loading summary: {e}")
        return None

def process_pointcloud_csv(csv_path: str, include_metadata: bool = True) -> pd.DataFrame:
    """
    Process a single pointcloud CSV: extract features per frame, add labels.
    
    Returns:
        DataFrame with features and label per row (one per frame)
    """
    print(f" Loading: {os.path.basename(csv_path)}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f" ⚠ Error loading file: {e}")
        return pd.DataFrame()
    
    if df.empty or 'frame_num' not in df.columns:
        print(f" ⚠ Empty or invalid dataframe")
        return pd.DataFrame()
    
    # Load ground truth (and optionally num_tracked_objects)
    summary_path = get_summary_file_path(csv_path)
    gt_df = load_ground_truth(summary_path)
    
    # Get unique frames
    unique_frames = sorted(df['frame_num'].unique())
    
    # Extract features for each frame
    all_features = []
    source_file = os.path.basename(csv_path) if include_metadata else None
    
    total_points_removed = 0
    total_points_original = 0
    
    for frame_num in unique_frames:
        frame_data = df[df['frame_num'] == frame_num]
        
        # Preprocess: apply filters
        filtered_data, original_count = preprocess_frame(frame_data)
        total_points_original += original_count
        total_points_removed += (original_count - len(filtered_data))
        
        # Compute features on filtered points
        feat_dict = extract_features_per_frame(filtered_data, original_count)
        
        # Add label and optional num_tracked_objects
        label = -1  # Default missing
        num_tracked = -1  # Default missing
        
        if gt_df is not None:
            gt_row = gt_df[gt_df['frame_num'] == frame_num]
            if not gt_row.empty:
                label = gt_row['ground_truth'].iloc[0]
                if ADD_TRACKED_OBJECTS_FEATURE and 'num_tracked_objects' in gt_row.columns:
                    num_tracked = gt_row['num_tracked_objects'].iloc[0]
        
        feat_dict['label'] = float(label)
        
        # Feature 2: Number of points detected as IN TARGET (from summary)
        if ADD_TRACKED_OBJECTS_FEATURE:
            feat_dict['num_tracked_objects'] = float(num_tracked)
        
        # Add metadata if requested
        if include_metadata:
            feat_dict['frame_num'] = float(frame_num)
            feat_dict['source_file'] = source_file
        
        all_features.append(feat_dict)
    
    if not all_features:
        return pd.DataFrame()
    
    features_df = pd.DataFrame(all_features)
    
    # Print filtering statistics
    if total_points_original > 0:
        rejection_pct = 100.0 * total_points_removed / total_points_original
        print(f" ✓ Processed {len(unique_frames)} frames from {os.path.basename(csv_path)}")
        if ENABLE_SPATIAL_FILTER or ENABLE_VELOCITY_FILTER or ENABLE_SNR_FILTER:
            print(f"   Filtered: {total_points_removed}/{total_points_original} points ({rejection_pct:.1f}%)")
    
    return features_df

def collect_files_by_split(input_dir: str, pattern: str = '_pointcloud') -> Dict[str, List[str]]:
    """
    Collect CSV files by split (train/val/test).
    """
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return {}
    
    files_by_split = {'train': [], 'val': [], 'test': []}
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.csv') and pattern in f]
    
    for filename in all_files:
        filepath = os.path.join(input_dir, filename)
        if '_train' in filename:
            files_by_split['train'].append(filepath)
        elif '_val' in filename:
            files_by_split['val'].append(filepath)
        elif '_test' in filename:
            files_by_split['test'].append(filepath)
    
    return files_by_split

def process_split_data(file_list: List[str]) -> pd.DataFrame:
    """
    Process all files for a split, combine into one DataFrame.
    """
    all_dfs = []
    for filepath in file_list:
        split_df = process_pointcloud_csv(filepath, include_metadata=INCLUDE_METADATA)
        if not split_df.empty:
            all_dfs.append(split_df)
    
    if not all_dfs:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

# =========================
# MAIN FUNCTION
# =========================

def main():
    """Main execution function."""
    print(f"\n{'='*80}")
    print(f"RADAR FEATURE EXTRACTOR FOR CLASSICAL ML")
    print(f"{'='*80}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nPREPROCESSING FILTERS:")
    print(f"  Spatial filter: {ENABLE_SPATIAL_FILTER}")
    if ENABLE_SPATIAL_FILTER:
        print(f"    X: [{SPATIAL_LIMITS['x_min']}, {SPATIAL_LIMITS['x_max']}]")
        print(f"    Y: [{SPATIAL_LIMITS['y_min']}, {SPATIAL_LIMITS['y_max']}]")
        print(f"    Z: [{SPATIAL_LIMITS['z_min']}, {SPATIAL_LIMITS['z_max']}]")
    print(f"  Velocity filter: {ENABLE_VELOCITY_FILTER}")
    if ENABLE_VELOCITY_FILTER:
        print(f"    Threshold: |v| >= {VELOCITY_THRESHOLD} m/s")
    print(f"  SNR filter: {ENABLE_SNR_FILTER}")
    if ENABLE_SNR_FILTER:
        print(f"    Threshold: SNR >= {SNR_THRESHOLD} dB")
    print(f"\nFEATURE OPTIONS:")
    print(f"  Add num_tracked_objects: {ADD_TRACKED_OBJECTS_FEATURE}")
    print(f"  Add filter statistics: {ADD_FILTER_STATS}")
    print(f"  Include metadata: {INCLUDE_METADATA}")
    print(f"{'='*80}\n")
    
    ensure_dir(OUTPUT_DIR)
    
    # Collect files
    print("Collecting input files...")
    files_by_split = collect_files_by_split(INPUT_DIR, '_pointcloud')
    for split, files in files_by_split.items():
        if files:
            print(f" {split}: {len(files)} file(s)")
    print()
    
    # Process each split
    total_frames = 0
    for split_name in ['train', 'val', 'test']:
        file_list = files_by_split.get(split_name, [])
        if not file_list:
            print(f"No {split_name} files found, skipping...\n")
            continue
        
        print(f"{'='*80}")
        print(f"Processing {split_name.upper()} split...")
        print(f"{'='*80}")
        
        data_df = process_split_data(file_list)
        if data_df.empty:
            print(f"⚠ No valid data for {split_name} split\n")
            continue
        
        # Save CSV
        output_path = os.path.join(OUTPUT_DIR, f'features_{split_name}.csv')
        data_df.to_csv(output_path, index=False)
        print(f"\n ✓ Saved: {output_path} ({len(data_df)} frames)")
        
        # Stats
        unique_labels = data_df['label'].unique()
        print(f" Unique labels in {split_name}: {sorted(unique_labels)}")
        if -1 in unique_labels:
            missing = (data_df['label'] == -1).sum()
            print(f" ⚠ Missing labels: {missing} frames")
        
        total_frames += len(data_df)
        print()
    
    # Save config
    config_path = os.path.join(OUTPUT_DIR, 'features_config.txt')
    with open(config_path, 'w') as f:
        f.write("Feature Extractor Configuration\n")
        f.write(f"{'='*80}\n")
        f.write(f"Features extracted: 29 statistical features\n")
        f.write(f"  - Computed on all detected points per frame (after filtering)\n")
        f.write(f"  - num_tracked_objects from summary: {ADD_TRACKED_OBJECTS_FEATURE}\n")
        f.write(f"  - Filter statistics: {ADD_FILTER_STATS}\n")
        f.write(f"\nPreprocessing Filters:\n")
        f.write(f"  Spatial filter: {ENABLE_SPATIAL_FILTER}\n")
        if ENABLE_SPATIAL_FILTER:
            f.write(f"    X: [{SPATIAL_LIMITS['x_min']}, {SPATIAL_LIMITS['x_max']}]\n")
            f.write(f"    Y: [{SPATIAL_LIMITS['y_min']}, {SPATIAL_LIMITS['y_max']}]\n")
            f.write(f"    Z: [{SPATIAL_LIMITS['z_min']}, {SPATIAL_LIMITS['z_max']}]\n")
        f.write(f"  Velocity filter: {ENABLE_VELOCITY_FILTER}\n")
        if ENABLE_VELOCITY_FILTER:
            f.write(f"    Threshold: |v| >= {VELOCITY_THRESHOLD} m/s\n")
        f.write(f"  SNR filter: {ENABLE_SNR_FILTER}\n")
        if ENABLE_SNR_FILTER:
            f.write(f"    Threshold: SNR >= {SNR_THRESHOLD} dB\n")
        f.write(f"\nTotal frames processed: {total_frames}\n")
        f.write(f"\nFeature List (in order):\n")
        if ADD_FILTER_STATS:
            f.write(f"  - num_points_before_filter\n")
            f.write(f"  - num_points_after_filter\n")
            f.write(f"  - filter_rejection_ratio\n")
        f.write(f"  1. num_total_points\n")
        if ADD_TRACKED_OBJECTS_FEATURE:
            f.write(f"  2. num_tracked_objects (from summary)\n")
        f.write(f"  3. mean_range, std_range\n")
        f.write(f"  4. mean_doppler, std_doppler\n")
        f.write(f"  5. mean_snr, std_snr\n")
        f.write(f"  6-20. X/Y/Z statistics (mean, std, min, max)\n")
        f.write(f"  21-23. Ratios (x_y, x_z, z_y)\n")
        f.write(f"  24-29. Magnitudes (range, doppler, snr, x, y, z)\n")
        f.write(f"  + label (ground_truth: 0, 1, 2, 3 people)\n")
    
    print(f"{'='*80}")
    print(f"✓ FEATURE EXTRACTION COMPLETE!")
    print(f" Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f" Configuration: {config_path}")
    print(f" Use the CSVs (e.g., features_train.csv) for Random Forest/SVM training.")
    print(f" Target: Classify 'label' column (0,1,2,3 people).")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
