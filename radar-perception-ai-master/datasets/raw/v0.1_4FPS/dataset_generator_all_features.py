# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:53:19 2025

@author: TELLCOM
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Modal Radar Dataset Generator (PointCloud + Tracks + Height)

This script processes processed CSV files and converts them into synchronized .npy files 
for machine learning. It handles three data streams:
1. Point Cloud
2. Track Data
3. Height Data

All streams are aligned by frame number and zero-padded to a fixed size (e.g., 64 items).
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Tuple, Optional

# =========================
# CONFIGURATION
# =========================

#PARSED_OUTPUT_DIR = "parsed_data_all_feature"

# ========== INPUT/OUTPUT PATHS ==========
INPUT_DIR =  "split_data_all_feature"  # Directory containing the split CSV files
OUTPUT_DIR = "processed_dataset_multimodal"  # Directory to save .npy files

# ========== ZERO-PADDING CONFIGURATION ==========
# Maximum number of items (points/tracks/objects) per frame
MAX_ITEMS_PER_FRAME = 64 
PADDING_VALUE = 0.0

# ========== FEATURE SELECTION ==========
# Define features to extract for each modality. 
# These must match the column names in your CSV files.

# 1. Point Cloud Features
FEAT_POINTCLOUD = ['x', 'y', 'z', 'velocity', 'snr']

# 2. Track Data Features
FEAT_TRACKS = ['x', 'y', 'z', 'vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z']

# 3. Height Data Features
FEAT_HEIGHT = ['height', 'z'] # Adjust based on your CSV columns (e.g., 'height', 'maxZ')

# ========== LABEL CONFIGURATION ==========
LABEL_COLUMN = 'ground_truth' # Column name in summary files

# =========================
# HELPER FUNCTIONS
# =========================

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def pad_or_truncate(array: np.ndarray, target_size: int) -> np.ndarray:
    """
    Pad with zeros or truncate an array to a fixed length.
    
    Args:
        array: Input array of shape (N, features)
        target_size: Desired size N
        
    Returns:
        Array of shape (target_size, features)
    """
    current_size, num_features = array.shape
    
    if current_size == target_size:
        return array
    
    if current_size < target_size:
        # Pad with zeros
        padding = np.zeros((target_size - current_size, num_features), dtype=array.dtype)
        return np.vstack([array, padding])
    else:
        # Truncate (take first N)
        return array[:target_size, :]

def get_file_sets(input_dir: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Groups related CSV files (PC, Tracks, Height, Summary) by split and recording name.
    
    Returns structure:
    {
        'train': [
            {'base': 'file1', 'pc': 'path...', 'track': 'path...', 'height': 'path...', 'sum': 'path...'},
            ...
        ],
        'val': [...],
        'test': [...]
    }
    """
    sets = {'train': [], 'val': [], 'test': []}
    
    # We use pointcloud files as the "anchor" to find others
    pc_files = glob.glob(os.path.join(input_dir, "*_pointcloud_*.csv"))
    
    for pc_path in pc_files:
        filename = os.path.basename(pc_path)
        
        # Determine split
        if '_train.csv' in filename:
            split = 'train'
            suffix = '_train.csv'
        elif '_val.csv' in filename:
            split = 'val'
            suffix = '_val.csv'
        elif '_test.csv' in filename:
            split = 'test'
            suffix = '_test.csv'
        else:
            continue
            
        # Reconstruct base name and sibling paths
        # Format expected: {BaseName}_pointcloud_{split}.csv
        base_name = filename.replace(f'_pointcloud{suffix}', '')
        
        track_path = os.path.join(input_dir, f"{base_name}_tracks{suffix}")
        height_path = os.path.join(input_dir, f"{base_name}_height{suffix}")
        
        # Summary usually has a slightly different naming convention from the splitter
        # It might be named {BaseName}_summary_{split}.csv or similar.
        # Assuming standard output from previous script:
        summary_path = os.path.join(input_dir, f"{base_name}_summary{suffix}")
        
        # Check if summary exists (crucial for labels)
        if not os.path.exists(summary_path):
            # Try fallback name (sometimes it's _all.csv or just .csv depending on splitter logic)
            summary_path = os.path.join(input_dir, f"{base_name}_summary_all.csv")

        file_set = {
            'base_name': base_name,
            'pc': pc_path,
            'tracks': track_path,
            'height': height_path,
            'summary': summary_path
        }
        sets[split].append(file_set)
        
    return sets

def load_csv_safe(path: str, required_cols: List[str]) -> pd.DataFrame:
    """Loads CSV, returns empty DF with correct columns if file missing or invalid."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=['frame_num'] + required_cols)
    
    try:
        df = pd.read_csv(path)
        # Ensure frame_num exists
        if 'frame_num' not in df.columns:
            return pd.DataFrame(columns=['frame_num'] + required_cols)
            
        # Ensure required features exist (fill missing with 0)
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        return df
    except Exception as e:
        print(f"    Warning: Error reading {os.path.basename(path)}: {e}")
        return pd.DataFrame(columns=['frame_num'] + required_cols)

def process_recording(file_set: Dict[str, str]) -> Tuple[Optional[np.ndarray], ...]:
    """
    Process one set of files (PC, Tracks, Height) corresponding to one recording.
    Returns tuple of (pc_arr, tr_arr, ht_arr, label_arr)
    """
    
    # 1. Load Summary (Ground Truth & Frame Master List)
    if not os.path.exists(file_set['summary']):
        print(f"  Skipping {file_set['base_name']}: Missing summary file")
        return None, None, None, None
        
    df_sum = pd.read_csv(file_set['summary'])
    if 'frame_num' not in df_sum.columns or LABEL_COLUMN not in df_sum.columns:
        return None, None, None, None
        
    # Get sorted unique frames from summary (this ensures alignment)
    unique_frames = sorted(df_sum['frame_num'].unique())
    num_frames = len(unique_frames)
    
    # 2. Load Data CSVs
    df_pc = load_csv_safe(file_set['pc'], FEAT_POINTCLOUD)
    df_tr = load_csv_safe(file_set['tracks'], FEAT_TRACKS)
    df_ht = load_csv_safe(file_set['height'], FEAT_HEIGHT)
    
    # 3. Pre-group dataframes by frame_num for O(1) access inside loop
    # This is much faster than filtering inside the loop
    gb_pc = df_pc.groupby('frame_num')
    gb_tr = df_tr.groupby('frame_num')
    gb_ht = df_ht.groupby('frame_num')
    
    # 4. Initialize Output Arrays
    # Shape: (Num_Frames, Max_Items, Num_Features)
    out_pc = np.zeros((num_frames, MAX_ITEMS_PER_FRAME, len(FEAT_POINTCLOUD)), dtype=np.float32)
    out_tr = np.zeros((num_frames, MAX_ITEMS_PER_FRAME, len(FEAT_TRACKS)), dtype=np.float32)
    out_ht = np.zeros((num_frames, MAX_ITEMS_PER_FRAME, len(FEAT_HEIGHT)), dtype=np.float32)
    out_lbl = np.zeros((num_frames,), dtype=np.int32)
    
    # 5. Iterate and Process
    for i, frame_num in enumerate(unique_frames):
        # --- Process Point Cloud ---
        if frame_num in gb_pc.groups:
            raw = gb_pc.get_group(frame_num)[FEAT_POINTCLOUD].values.astype(np.float32)
            out_pc[i] = pad_or_truncate(raw, MAX_ITEMS_PER_FRAME)
            
        # --- Process Tracks ---
        if frame_num in gb_tr.groups:
            raw = gb_tr.get_group(frame_num)[FEAT_TRACKS].values.astype(np.float32)
            out_tr[i] = pad_or_truncate(raw, MAX_ITEMS_PER_FRAME)
            
        # --- Process Height ---
        if frame_num in gb_ht.groups:
            raw = gb_ht.get_group(frame_num)[FEAT_HEIGHT].values.astype(np.float32)
            out_ht[i] = pad_or_truncate(raw, MAX_ITEMS_PER_FRAME)
            
        # --- Process Labels ---
        # Assuming one label per frame in summary
        label_val = df_sum.loc[df_sum['frame_num'] == frame_num, LABEL_COLUMN].iloc[0]
        out_lbl[i] = int(label_val)
        
    return out_pc, out_tr, out_ht, out_lbl

# =========================
# MAIN
# =========================

def main():
    print(f"\n{'='*80}")
    print(f"MULTIMODAL RADAR DATASET GENERATOR")
    print(f"{'='*80}")
    print(f"Config: Max Items per Frame = {MAX_ITEMS_PER_FRAME}")
    print(f"Features PC    : {len(FEAT_POINTCLOUD)} dims {FEAT_POINTCLOUD}")
    print(f"Features Tracks: {len(FEAT_TRACKS)} dims {FEAT_TRACKS}")
    print(f"Features Height: {len(FEAT_HEIGHT)} dims {FEAT_HEIGHT}")
    print(f"{'='*80}\n")
    
    ensure_dir(OUTPUT_DIR)
    
    # Get all file sets
    file_sets = get_file_sets(INPUT_DIR)
    
    for split_name, recordings in file_sets.items():
        if not recordings:
            continue
            
        print(f"Processing {split_name.upper()} split ({len(recordings)} recordings)...")
        
        list_pc, list_tr, list_ht, list_lbl = [], [], [], []
        
        for rec in recordings:
            print(f"  - {rec['base_name']}")
            pc, tr, ht, lbl = process_recording(rec)
            
            if pc is not None:
                list_pc.append(pc)
                list_tr.append(tr)
                list_ht.append(ht)
                list_lbl.append(lbl)
                
        if not list_pc:
            print(f"  ⚠ No valid data found for {split_name}")
            continue
            
        # Concatenate all recordings for this split
        final_pc = np.concatenate(list_pc, axis=0)
        final_tr = np.concatenate(list_tr, axis=0)
        final_ht = np.concatenate(list_ht, axis=0)
        final_lbl = np.concatenate(list_lbl, axis=0)
        
        # Save to NPY
        print(f"  > Saving {split_name} data...")
        
        np.save(os.path.join(OUTPUT_DIR, f'{split_name}_pointcloud.npy'), final_pc)
        np.save(os.path.join(OUTPUT_DIR, f'{split_name}_tracks.npy'), final_tr)
        np.save(os.path.join(OUTPUT_DIR, f'{split_name}_height.npy'), final_ht)
        np.save(os.path.join(OUTPUT_DIR, f'{split_name}_labels.npy'), final_lbl)
        
        print(f"    Final Shapes:")
        print(f"    PC:     {final_pc.shape}")
        print(f"    Tracks: {final_tr.shape}")
        print(f"    Height: {final_ht.shape}")
        print(f"    Labels: {final_lbl.shape}")
        print("-" * 40)

    print(f"\n✓ Done! Output saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()