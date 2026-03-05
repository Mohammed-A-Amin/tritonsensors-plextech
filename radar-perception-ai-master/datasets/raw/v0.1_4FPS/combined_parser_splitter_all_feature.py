# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:37:58 2025

@author: TELLCOM
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Radar Data Processor - Parse JSON and Split into Train/Val/Test
Includes: PointCloud, TrackData, and HeightData
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# =========================
# CONFIGURATION (edit here)
# =========================

# ========== OPERATION MODE ==========
RUN_PARSER = True   # Convert JSON to CSV
RUN_SPLITTER = True # Split CSV into train/val/test

# ========== PARSER CONFIGURATION ==========
INPUT_FILES_AND_LABELS = [
    (r"ParsaOffice_TwoPeople_CloseInteraction_T0.json", "2_person"),
    (r"ParsaOffice_TwoPeople_Occlusion_T0.json", "2_person"),
    (r"ParsaOffice_TwoPeople_RandomWalk_T0.json", "2_person"),
    (r"ParsaOffice_TwoPeople_SpacedApart_T0.json", "2_person"),
    (r"ParsaOffice_TwoPeople_StaticMover_T0.json", "2_person"),
    (r"ParsaOffice_TwoPeople_CloseInteraction_T1.json", "2_person"),
    (r"ParsaOffice_TwoPeople_Occlusion_T1.json", "2_person"),
    (r"ParsaOffice_TwoPeople_RandomWalk_T1.json", "2_person"),
    (r"ParsaOffice_TwoPeople_SpaceApart_T1.json", "2_person"),
    (r"ParsaOffice_TwoPeople_StaticMover_T1.json", "2_person"),
    (r"ParsaOffice_EmptyRoom_T0.json", "0_person"),
    (r"ParsaOffice_EmptyRoom_T1.json", "0_person"),
    (r"ParsaOffice_SinglePerson_NearWall_T0.json", "1_person"),
    (r"ParsaOffice_SinglePerson_RandomWalk_T0.json", "1_person"),
    (r"ParsaOffice_SinglePerson_SitStand_T0.json", "1_person"),
    (r"ParsaOffice_SinglePerson_StandingCorners_T0.json", "1_person"),
    (r"ParsaOffice_SinglePerson_StraightLine_T0.json", "1_person"),
    (r"ParsaOffice_ThreePeople_WalkinLine_T0.json", "3_person"),
    (r"ParsaOffice_ThreePeople_GroupingDispersing_T0.json", "3_person"),
    (r"ParsaOffice_ThreePeople_RandomWalk_T0.json", "3_person"),
]


PARSED_OUTPUT_DIR = "parsed_data_all_feature"
SPLIT_OUTPUT_DIR = "split_data_all_feature"

# Frame filtering
SKIP_FIRST = 0
SKIP_LAST = 0

# Radar configuration
RADAR_HEIGHT_M = 2.7
VERTICAL_AXIS = "y"
APPLY_FLOOR_ELEVATION = True

# Parser output options
CREATE_POINTCLOUD_CSV = True
CREATE_TRACK_CSV = True
CREATE_HEIGHT_CSV = True 
CREATE_SUMMARY_CSV = True

# ========== SPLITTER CONFIGURATION ==========
TRAIN_SPLIT_PERCENT = 0.7
VAL_SPLIT_PERCENT = 0.0
TEST_SPLIT_PERCENT = 0.3

# Splitter output options
CREATE_SEPARATE_FILES = True   # _train.csv, _val.csv, _test.csv
CREATE_COMBINED_FILE = True    # _all.csv with 'split' column

# =========================
# PARSER FUNCTIONS
# =========================

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def parse_people_label(lbl: str) -> int:
    """Parse labels like '1_person', '2_person' to integer counts."""
    if lbl is None:
        raise ValueError("Ground-truth label is None.")
    s = str(lbl).strip().lower()
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        elif num:
            break
    if not num:
        raise ValueError(f"Could not parse integer people count from label: {lbl}")
    return int(num)

def load_radar_json_file(path: str) -> Optional[Dict[str, Any]]:
    """Load JSON radar data file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data or "data" not in data:
            return None
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def filter_edge_frames(n_frames: int, start_skip: int, end_skip: int) -> List[int]:
    """Returns list of frame indices to keep after edge filtering."""
    if n_frames == 0:
        return []
    start_idx = max(0, int(start_skip))
    end_idx = n_frames - max(0, int(end_skip))
    if start_idx >= end_idx:
        mid = n_frames // 2
        return list(range(max(0, mid - 1), min(mid + 1, n_frames)))
    return list(range(start_idx, end_idx))

def process_point_cloud(frame_data: Dict[str, Any], frame_num: int) -> pd.DataFrame:
    """Process point cloud data for a single frame."""
  # Get data considering the default value of empty list
    raw = np.array(frame_data.get("pointCloud", []), dtype=float)

  # If data was empty or did not exist (as requested: consider 0)
    if raw.size == 0:
     # Return a row with zero values ​​so the frame is not empty
        return pd.DataFrame({
            "frame_num": [frame_num],
            "x": [0.0], "y": [0.0], "z": [0.0], 
            "velocity": [0.0], "snr": [0.0], "noise": [0.0], "track_idx": [0]
        })

    # If the points were in flat list format (no columns), they should be converted.
    # The standard format is assumed to be a list of lists or a 2D array.
    try:
        df = pd.DataFrame(raw, columns=["x", "y", "z", "velocity", "snr", "noise", "track_idx"])
    except ValueError:
     # If the data dimensions do not match the columns
        df = pd.DataFrame(raw)
        df.columns = ["x", "y", "z", "velocity", "snr", "noise", "track_idx"][:len(df.columns)]

    if APPLY_FLOOR_ELEVATION and VERTICAL_AXIS in df.columns:
        df[VERTICAL_AXIS] = RADAR_HEIGHT_M - df[VERTICAL_AXIS].astype(float)

    df.insert(0, "frame_num", frame_num)
    return df

def process_track_data(frame_data: Dict[str, Any], frame_num: int) -> pd.DataFrame:
    """Process track data for a single frame."""
    track_data = frame_data.get("trackData", [])

    # If trackdata did not exist or was empty, we return a zero row
    if not track_data:
        return pd.DataFrame({
            "frame_num": [frame_num],
            "track_id": [0], "x": [0.0], "y": [0.0], "z": [0.0], 
            "vel_x": [0.0], "vel_y": [0.0], "vel_z": [0.0], 
            "acc_x": [0.0], "acc_y": [0.0], "acc_z": [0.0]
        })

    df_tracks = pd.DataFrame(track_data)

    # Mapping column names (if names are different)
    expected_columns = [
        "tid", "posX", "posY", "posZ", "velX", "velY", "velZ",
        "accX", "accY", "accZ", "ec1", "ec2", "ec3", "ec4", "ec5", "ec6"
    ]
    
    # If the columns are unnamed and are index-based only
    if isinstance(track_data[0], list):
         df_tracks.columns = expected_columns[:len(df_tracks.columns)]
    
    # Apply height settings
    if "posY" in df_tracks.columns and APPLY_FLOOR_ELEVATION:
        df_tracks["posY"] = RADAR_HEIGHT_M - df_tracks["posY"]

    # Select final columns
    rename_map = {
        "tid": "track_id",
        "posX": "x", "posY": "y", "posZ": "z",
        "velX": "vel_x", "velY": "vel_y", "velZ": "vel_z",
        "accX": "acc_x", "accY": "acc_y", "accZ": "acc_z"
    }
    df_tracks.rename(columns=rename_map, inplace=True)
    
    final_cols = ["track_id", "x", "y", "z", "vel_x", "vel_y", "vel_z", "acc_x", "acc_y", "acc_z"]
    # We only keep columns that exist
    cols_to_keep = [c for c in final_cols if c in df_tracks.columns]
    
    df_tracks = df_tracks[cols_to_keep].copy()
    df_tracks.insert(0, "frame_num", frame_num)

    return df_tracks

def process_height_data(frame_data: Dict[str, Any], frame_num: int) -> pd.DataFrame:
    """Process height data for a single frame."""
    # Get HeightData data, if not present returns an empty list
    height_data = frame_data.get("heightData", [])

    # If empty (or None), create a row with zero values
    if not height_data:
        return pd.DataFrame({
            "frame_num": [frame_num],
            "track_id": [0],   # It is assumed that heightData is usually associated with trackId
            "height": [0.0],
            "z": [0.0]         # Sometimes heightData also includes Z
        })

    df_height = pd.DataFrame(height_data)
    
    # Attempt to standardize column names
    # Usually heightData contains TID and height value
    rename_map = {
        "tid": "track_id",
        "oid": "track_id",   # Sometimes called object id
        "height": "height",
        "maxZ": "height_max",
        "minZ": "height_min"
    }
    df_height.rename(columns=rename_map, inplace=True)
    
    # If columns are not recognized, leave as is
    df_height.insert(0, "frame_num", frame_num)
    
    return df_height

def get_frame_summary(frame_data: Dict[str, Any], frame_num: int, 
                      point_count: int, track_count: int, gt_count: int) -> Dict[str, Any]:
    """Create summary statistics for a frame."""
    # Actual number of data (excluding dummy zero rows)
    # Note: Because in the above functions we added a zero row for empty data,
    # Here we need to check if it was really data or not.
    # But for simplicity we write the length of the output data.    
    pc_len = len(frame_data.get("pointCloud", []) or [])
    tr_len = len(frame_data.get("trackData", []) or [])
    
    return {
        "frame_num": frame_num,
        "num_points": pc_len,
        "num_tracked_objects": tr_len,
        "ground_truth": gt_count
    }

def run_parser() -> int:
    """Run the JSON to CSV parser."""
    ensure_dir(PARSED_OUTPUT_DIR)

    print(f"\n{'='*80}")
    print(f"STEP 1: PARSING JSON TO CSV (Including HeightData)")
    print(f"{'='*80}")
    
    files_processed = 0

    for file_path, gt_label in INPUT_FILES_AND_LABELS:
        print(f"\nProcessing: {os.path.basename(file_path)}")

        radar_data = load_radar_json_file(file_path)
        if radar_data is None:
            continue

        try:
            gt_count = parse_people_label(gt_label)
        except ValueError:
            print(f"  Warning: Could not parse label '{gt_label}', assuming 0.")
            gt_count = 0

        all_frames = radar_data.get("data", [])
        total_frames = len(all_frames)

        if total_frames == 0:
            continue

        kept_indices = filter_edge_frames(total_frames, SKIP_FIRST, SKIP_LAST)
        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        all_pointcloud_data = []
        all_track_data = []
        all_height_data = []  # لیست جدید
        all_summary_data = []

        for local_idx in kept_indices:
            frame = all_frames[local_idx]
            frame_data = frame.get("frameData", {})
            # If frameData does not exist, pass an empty dictionary so that the functions below will return 0.
            if frame_data is None: 
                frame_data = {}
                
            frame_num = frame_data.get("frameNum", local_idx)

            if CREATE_POINTCLOUD_CSV:
                df_points = process_point_cloud(frame_data, frame_num)
                all_pointcloud_data.append(df_points)

            if CREATE_TRACK_CSV:
                df_tracks = process_track_data(frame_data, frame_num)
                all_track_data.append(df_tracks)
            
            if CREATE_HEIGHT_CSV:
                df_height = process_height_data(frame_data, frame_num)
                all_height_data.append(df_height)

            if CREATE_SUMMARY_CSV:
                # Counts for summary are based on RAW json to avoid counting dummy rows
                raw_pc = frame_data.get("pointCloud", [])
                pt_cnt = len(raw_pc) if raw_pc else 0
                
                raw_tr = frame_data.get("trackData", [])
                tr_cnt = len(raw_tr) if raw_tr else 0
                
                summary = get_frame_summary(frame_data, frame_num, pt_cnt, tr_cnt, gt_count)
                all_summary_data.append(summary)

        # --- SAVE PointCloud ---
        if CREATE_POINTCLOUD_CSV and all_pointcloud_data:
            df_all = pd.concat(all_pointcloud_data, ignore_index=True)
            path = os.path.join(PARSED_OUTPUT_DIR, f"{base_filename}_pointcloud.csv")
            df_all.to_csv(path, index=False)
            print(f"  ✓ Saved PointCloud: {len(df_all)} rows")

        # --- SAVE Tracks ---
        if CREATE_TRACK_CSV and all_track_data:
            df_all = pd.concat(all_track_data, ignore_index=True)
            path = os.path.join(PARSED_OUTPUT_DIR, f"{base_filename}_tracks.csv")
            df_all.to_csv(path, index=False)
            print(f"  ✓ Saved Tracks: {len(df_all)} rows")

        # --- SAVE HeightData (NEW) ---
        if CREATE_HEIGHT_CSV and all_height_data:
            df_all = pd.concat(all_height_data, ignore_index=True)
            path = os.path.join(PARSED_OUTPUT_DIR, f"{base_filename}_height.csv")
            df_all.to_csv(path, index=False)
            print(f"  ✓ Saved HeightData: {len(df_all)} rows")

        # --- SAVE Summary ---
        if CREATE_SUMMARY_CSV and all_summary_data:
            df_all = pd.DataFrame(all_summary_data)
            path = os.path.join(PARSED_OUTPUT_DIR, f"{base_filename}_summary.csv")
            df_all.to_csv(path, index=False)

        files_processed += 1

    return files_processed

# =========================
# SPLITTER FUNCTIONS
# =========================

def validate_split_config() -> None:
    total = TRAIN_SPLIT_PERCENT + VAL_SPLIT_PERCENT + TEST_SPLIT_PERCENT
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Split percentages must sum to 1.0 (Currently: {total:.2f})")

def get_csv_files(directory: str, pattern: str) -> List[str]:
    if not os.path.exists(directory):
        return []
    all_files = os.listdir(directory)
    matching_files = [f for f in all_files if f.endswith(pattern)]
    return sorted(matching_files)

def split_dataframe(df: pd.DataFrame, train_pct: float, val_pct: float, 
                   test_pct: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()

    unique_frames = sorted(df['frame_num'].unique())
    n_frames = len(unique_frames)

    train_end = int(n_frames * train_pct)
    val_end = train_end + int(n_frames * val_pct)

    train_frames = unique_frames[:train_end]
    val_frames = unique_frames[train_end:val_end]
    test_frames = unique_frames[val_end:]

    df_train = df[df['frame_num'].isin(train_frames)].copy()
    df_val = df[df['frame_num'].isin(val_frames)].copy()
    df_test = df[df['frame_num'].isin(test_frames)].copy()

    return df_train, df_val, df_test

def create_combined_dataframe(df_train, df_val, df_test):
    dfs = []
    if not df_train.empty:
        d = df_train.copy()
        d['split'] = 'train'
        dfs.append(d)
    if not df_val.empty:
        d = df_val.copy()
        d['split'] = 'val'
        dfs.append(d)
    if not df_test.empty:
        d = df_test.copy()
        d['split'] = 'test'
        dfs.append(d)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def process_csv_file(file_path: str, output_base_name: str) -> Dict[str, Any]:
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return {"success": False}

    if df.empty or 'frame_num' not in df.columns:
        return {"success": False}

    df_train, df_val, df_test = split_dataframe(
        df, TRAIN_SPLIT_PERCENT, VAL_SPLIT_PERCENT, TEST_SPLIT_PERCENT
    )

    results = {
        "success": True,
        "files_created": []
    }

    # Save separate files
    if CREATE_SEPARATE_FILES:
        if not df_train.empty:
            p = os.path.join(SPLIT_OUTPUT_DIR, f"{output_base_name}_train.csv")
            df_train.to_csv(p, index=False)
            results["files_created"].append(p)
        if not df_val.empty:
            p = os.path.join(SPLIT_OUTPUT_DIR, f"{output_base_name}_val.csv")
            df_val.to_csv(p, index=False)
            results["files_created"].append(p)
        if not df_test.empty:
            p = os.path.join(SPLIT_OUTPUT_DIR, f"{output_base_name}_test.csv")
            df_test.to_csv(p, index=False)
            results["files_created"].append(p)

    # Save combined file
    if CREATE_COMBINED_FILE:
        df_comb = create_combined_dataframe(df_train, df_val, df_test)
        if not df_comb.empty:
            p = os.path.join(SPLIT_OUTPUT_DIR, f"{output_base_name}_all.csv")
            df_comb.to_csv(p, index=False)
            results["files_created"].append(p)

    return results

def run_splitter() -> Tuple[int, int]:
    print(f"\n{'='*80}")
    print(f"STEP 2: SPLITTING CSV DATA")
    print(f"{'='*80}")

    validate_split_config()
    ensure_dir(SPLIT_OUTPUT_DIR)

    if not os.path.exists(PARSED_OUTPUT_DIR):
        print("Parsed directory not found.")
        return 0, 0

    # Define what files to look for
    csv_patterns = []
    if CREATE_POINTCLOUD_CSV:
        csv_patterns.append(("_pointcloud.csv", "Point Cloud"))
    if CREATE_TRACK_CSV:
        csv_patterns.append(("_tracks.csv", "Track"))
    if CREATE_HEIGHT_CSV:
        csv_patterns.append(("_height.csv", "Height Data")) # <--- اضافه شده
    if CREATE_SUMMARY_CSV:
        csv_patterns.append(("_summary.csv", "Summary"))

    total_processed = 0
    total_created = 0

    for pattern, label in csv_patterns:
        print(f"\n--- Processing {label} Files ---")
        files = get_csv_files(PARSED_OUTPUT_DIR, pattern)
        for f in files:
            path = os.path.join(PARSED_OUTPUT_DIR, f)
            base = f.replace(".csv", "")
            res = process_csv_file(path, base)
            if res["success"]:
                total_processed += 1
                total_created += len(res["files_created"])
                print(f"  ✓ Split: {f}")

    return total_processed, total_created

# =========================
# MAIN FUNCTION
# =========================

def main():
    if RUN_PARSER:
        run_parser()
    if RUN_SPLITTER:
        run_splitter()
    print("\nDone.")

if __name__ == "__main__":
    main()