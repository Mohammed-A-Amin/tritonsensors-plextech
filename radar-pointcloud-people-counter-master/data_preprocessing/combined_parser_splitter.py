#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Radar Data Processor - Parse JSON and Split into Train/Val/Test
Can run both operations together or separately
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
RUN_SPLITTER = True  # Split CSV into train/val/test

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

PARSED_OUTPUT_DIR = "parsed_data"
SPLIT_OUTPUT_DIR = "split_data"

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
    raw = np.array(frame_data.get("pointCloud", []), dtype=float)

    if raw.size == 0:
        return pd.DataFrame(columns=["frame_num", "x", "y", "z", "velocity", 
                                     "snr", "noise", "track_idx"])

    df = pd.DataFrame(raw, columns=["x", "y", "z", "velocity", "snr", "noise", "track_idx"])

    if APPLY_FLOOR_ELEVATION and VERTICAL_AXIS in df.columns:
        df[VERTICAL_AXIS] = RADAR_HEIGHT_M - df[VERTICAL_AXIS].astype(float)

    df.insert(0, "frame_num", frame_num)
    return df

def process_track_data(frame_data: Dict[str, Any], frame_num: int) -> pd.DataFrame:
    """Process track data for a single frame."""
    track_data = frame_data.get("trackData", [])

    if not track_data:
        return pd.DataFrame(columns=["frame_num", "track_id", "x", "y", "z", 
                                     "vel_x", "vel_y", "vel_z", "acc_x", "acc_y", "acc_z"])

    df_tracks = pd.DataFrame(track_data)

    expected_columns = [
        "tid", "posX", "posY", "posZ", "velX", "velY", "velZ",
        "accX", "accY", "accZ", "ec1", "ec2", "ec3", "ec4", "ec5", "ec6"
    ]

    df_tracks.columns = expected_columns[:len(df_tracks.columns)]

    if "posY" in df_tracks.columns:
        df_tracks["posY"] = RADAR_HEIGHT_M - df_tracks["posY"]

    columns_to_keep = ["tid", "posX", "posY", "posZ", "velX", "velY", "velZ", 
                       "accX", "accY", "accZ"]
    available_cols = [col for col in columns_to_keep if col in df_tracks.columns]
    df_tracks = df_tracks[available_cols].copy()

    rename_map = {
        "tid": "track_id",
        "posX": "x", "posY": "y", "posZ": "z",
        "velX": "vel_x", "velY": "vel_y", "velZ": "vel_z",
        "accX": "acc_x", "accY": "acc_y", "accZ": "acc_z"
    }
    df_tracks.rename(columns=rename_map, inplace=True)
    df_tracks.insert(0, "frame_num", frame_num)

    return df_tracks

def get_frame_summary(frame_data: Dict[str, Any], frame_num: int, 
                      point_count: int, track_count: int, gt_count: int) -> Dict[str, Any]:
    """Create summary statistics for a frame."""
    return {
        "frame_num": frame_num,
        "num_points": point_count,
        "num_tracked_objects": track_count,
        "ground_truth": gt_count
    }

def run_parser() -> int:
    """Run the JSON to CSV parser."""
    ensure_dir(PARSED_OUTPUT_DIR)

    print(f"\n{'='*80}")
    print(f"STEP 1: PARSING JSON TO CSV")
    print(f"{'='*80}")
    print(f"Processing {len(INPUT_FILES_AND_LABELS)} file(s)...")
    print(f"Output directory: {PARSED_OUTPUT_DIR}")
    print(f"{'='*80}\n")

    files_processed = 0

    for file_path, gt_label in INPUT_FILES_AND_LABELS:
        print(f"\nProcessing: {os.path.basename(file_path)} (GT: {gt_label})")

        radar_data = load_radar_json_file(file_path)
        if radar_data is None:
            print(f"  ⚠ Skipping invalid file: {file_path}")
            continue

        try:
            gt_count = parse_people_label(gt_label)
        except ValueError as e:
            print(f"  ⚠ Skipping file with invalid label: {e}")
            continue

        all_frames = radar_data.get("data", [])
        total_frames = len(all_frames)

        if total_frames == 0:
            print(f"  ⚠ No frames found in file")
            continue

        kept_indices = filter_edge_frames(total_frames, SKIP_FIRST, SKIP_LAST)
        print(f"  Total frames: {total_frames}")
        print(f"  Kept frames: {len(kept_indices)} (after filtering)")

        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        all_pointcloud_data = []
        all_track_data = []
        all_summary_data = []

        for local_idx in kept_indices:
            frame = all_frames[local_idx]
            frame_data = frame.get("frameData", {})
            frame_num = frame_data.get("frameNum", local_idx)

            if CREATE_POINTCLOUD_CSV:
                df_points = process_point_cloud(frame_data, frame_num)
                all_pointcloud_data.append(df_points)

            if CREATE_TRACK_CSV:
                df_tracks = process_track_data(frame_data, frame_num)
                all_track_data.append(df_tracks)

            if CREATE_SUMMARY_CSV:
                point_count = len(df_points) if CREATE_POINTCLOUD_CSV else 0
                track_count = len(df_tracks) if CREATE_TRACK_CSV else 0
                summary = get_frame_summary(frame_data, frame_num, 
                                           point_count, track_count, gt_count)
                all_summary_data.append(summary)

        if CREATE_POINTCLOUD_CSV and all_pointcloud_data:
            df_all_points = pd.concat(all_pointcloud_data, ignore_index=True)
            pointcloud_path = os.path.join(PARSED_OUTPUT_DIR, f"{base_filename}_pointcloud.csv")
            df_all_points.to_csv(pointcloud_path, index=False)
            print(f"  ✓ Saved: {pointcloud_path} ({len(df_all_points)} points)")

        if CREATE_TRACK_CSV and all_track_data:
            df_all_tracks = pd.concat(all_track_data, ignore_index=True)
            if not df_all_tracks.empty:
                track_path = os.path.join(PARSED_OUTPUT_DIR, f"{base_filename}_tracks.csv")
                df_all_tracks.to_csv(track_path, index=False)
                print(f"  ✓ Saved: {track_path} ({len(df_all_tracks)} tracks)")

        if CREATE_SUMMARY_CSV and all_summary_data:
            df_summary = pd.DataFrame(all_summary_data)
            summary_path = os.path.join(PARSED_OUTPUT_DIR, f"{base_filename}_summary.csv")
            df_summary.to_csv(summary_path, index=False)
            print(f"  ✓ Saved: {summary_path} ({len(df_summary)} frames)")

        files_processed += 1

    print(f"\n{'='*80}")
    print(f"✓ PARSING COMPLETE!")
    print(f"  Processed: {files_processed} file(s)")
    print(f"  Output directory: {os.path.abspath(PARSED_OUTPUT_DIR)}")
    print(f"{'='*80}\n")

    return files_processed

# =========================
# SPLITTER FUNCTIONS
# =========================

def validate_split_config() -> None:
    """Validate split configuration."""
    total = TRAIN_SPLIT_PERCENT + VAL_SPLIT_PERCENT + TEST_SPLIT_PERCENT
    if total < 0 or total > 1.0:
        raise ValueError(f"Train/Val/Test splits sum to {total:.2f}, must be in [0, 1.0]")

def get_csv_files(directory: str, pattern: str) -> List[str]:
    """Get all CSV files matching a pattern."""
    if not os.path.exists(directory):
        return []

    all_files = os.listdir(directory)
    matching_files = [f for f in all_files if f.endswith(pattern)]
    return sorted(matching_files)

def split_dataframe(df: pd.DataFrame, train_pct: float, val_pct: float, 
                   test_pct: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe by frame numbers into train/val/test sets."""
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()

    unique_frames = sorted(df['frame_num'].unique())
    n_frames = len(unique_frames)

    train_end = int(n_frames * train_pct)
    val_end = train_end + int(n_frames * val_pct)
    test_end = val_end + int(n_frames * test_pct)

    train_frames = unique_frames[:train_end]
    val_frames = unique_frames[train_end:val_end]
    test_frames = unique_frames[val_end:test_end]

    df_train = df[df['frame_num'].isin(train_frames)].copy()
    df_val = df[df['frame_num'].isin(val_frames)].copy()
    df_test = df[df['frame_num'].isin(test_frames)].copy()

    return df_train, df_val, df_test

def create_combined_dataframe(df_train: pd.DataFrame, df_val: pd.DataFrame, 
                              df_test: pd.DataFrame) -> pd.DataFrame:
    """Combine train/val/test dataframes with 'split' column."""
    if not df_train.empty:
        df_train = df_train.copy()
        df_train['split'] = 'train'
    if not df_val.empty:
        df_val = df_val.copy()
        df_val['split'] = 'val'
    if not df_test.empty:
        df_test = df_test.copy()
        df_test['split'] = 'test'

    dfs_to_combine = [df for df in [df_train, df_val, df_test] if not df.empty]

    if not dfs_to_combine:
        return pd.DataFrame()

    df_combined = pd.concat(dfs_to_combine, ignore_index=True)
    return df_combined

def process_csv_file(file_path: str, output_base_name: str) -> Dict[str, Any]:
    """Process a single CSV file and split it."""
    print(f"  Processing: {os.path.basename(file_path)}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"    ⚠ Error loading file: {e}")
        return {"success": False}

    if df.empty:
        print(f"    ⚠ Empty dataframe")
        return {"success": False}

    if 'frame_num' not in df.columns:
        print(f"    ⚠ No 'frame_num' column found")
        return {"success": False}

    df_train, df_val, df_test = split_dataframe(
        df, TRAIN_SPLIT_PERCENT, VAL_SPLIT_PERCENT, TEST_SPLIT_PERCENT
    )

    results = {
        "success": True,
        "total_rows": len(df),
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "test_rows": len(df_test),
        "files_created": []
    }

    if CREATE_SEPARATE_FILES:
        if not df_train.empty:
            train_path = os.path.join(SPLIT_OUTPUT_DIR, f"{output_base_name}_train.csv")
            df_train.to_csv(train_path, index=False)
            results["files_created"].append(train_path)

        if not df_val.empty:
            val_path = os.path.join(SPLIT_OUTPUT_DIR, f"{output_base_name}_val.csv")
            df_val.to_csv(val_path, index=False)
            results["files_created"].append(val_path)

        if not df_test.empty:
            test_path = os.path.join(SPLIT_OUTPUT_DIR, f"{output_base_name}_test.csv")
            df_test.to_csv(test_path, index=False)
            results["files_created"].append(test_path)

    if CREATE_COMBINED_FILE:
        df_combined = create_combined_dataframe(df_train, df_val, df_test)
        if not df_combined.empty:
            combined_path = os.path.join(SPLIT_OUTPUT_DIR, f"{output_base_name}_all.csv")
            df_combined.to_csv(combined_path, index=False)
            results["files_created"].append(combined_path)

    print(f"    ✓ Split: {results['train_rows']} train, "
          f"{results['val_rows']} val, {results['test_rows']} test")

    return results

def run_splitter() -> Tuple[int, int]:
    """Run the CSV data splitter."""
    print(f"\n{'='*80}")
    print(f"STEP 2: SPLITTING CSV DATA")
    print(f"{'='*80}")

    validate_split_config()

    if not os.path.exists(PARSED_OUTPUT_DIR):
        print(f"\n❌ Input directory not found: {PARSED_OUTPUT_DIR}")
        print(f"Please run parser first or check the directory path.")
        return 0, 0

    ensure_dir(SPLIT_OUTPUT_DIR)
    print(f"  Split ratios: Train={TRAIN_SPLIT_PERCENT*100:.0f}%, "
          f"Val={VAL_SPLIT_PERCENT*100:.0f}%, Test={TEST_SPLIT_PERCENT*100:.0f}%")
    print(f"  Input directory: {PARSED_OUTPUT_DIR}")
    print(f"  Output directory: {SPLIT_OUTPUT_DIR}")
    print(f"{'='*80}\n")

    total_files_processed = 0
    total_files_created = 0

    # Process all CSV types
    csv_patterns = []
    if CREATE_POINTCLOUD_CSV:
        csv_patterns.append(("_pointcloud.csv", "Point Cloud"))
    if CREATE_TRACK_CSV:
        csv_patterns.append(("_tracks.csv", "Track"))
    if CREATE_SUMMARY_CSV:
        csv_patterns.append(("_summary.csv", "Summary"))

    for pattern, label in csv_patterns:
        print(f"\n--- Processing {label} Files ---")
        csv_files = get_csv_files(PARSED_OUTPUT_DIR, pattern)
        print(f"Found {len(csv_files)} file(s)")

        for csv_file in csv_files:
            file_path = os.path.join(PARSED_OUTPUT_DIR, csv_file)
            base_name = csv_file.replace(".csv", "")

            result = process_csv_file(file_path, base_name)
            if result["success"]:
                total_files_processed += 1
                total_files_created += len(result["files_created"])

    print(f"\n{'='*80}")
    print(f"✓ SPLITTING COMPLETE!")
    print(f"  Processed: {total_files_processed} source file(s)")
    print(f"  Created: {total_files_created} output file(s)")
    print(f"  Output directory: {os.path.abspath(SPLIT_OUTPUT_DIR)}")
    print(f"{'='*80}\n")

    return total_files_processed, total_files_created

# =========================
# MAIN FUNCTION
# =========================

def main():
    """Main execution function."""
    print(f"\n{'='*80}")
    print(f"RADAR DATA PROCESSOR - COMBINED PIPELINE")
    print(f"{'='*80}")
    print(f"Mode: ", end="")

    if RUN_PARSER and RUN_SPLITTER:
        print("FULL PIPELINE (Parse + Split)")
    elif RUN_PARSER:
        print("PARSER ONLY")
    elif RUN_SPLITTER:
        print("SPLITTER ONLY")
    else:
        print("NO OPERATIONS SELECTED")
        return

    print(f"{'='*80}\n")

    try:
        # Run parser
        if RUN_PARSER:
            files_parsed = run_parser()
            if files_parsed == 0:
                print("⚠ No files were parsed successfully")
                if RUN_SPLITTER:
                    print("Skipping splitter due to parse failure")
                    return

        # Run splitter
        if RUN_SPLITTER:
            processed, created = run_splitter()
            if processed == 0:
                print("⚠ No files were split successfully")

        # Final summary
        print(f"\n{'='*80}")
        print(f"✓ ALL OPERATIONS COMPLETE!")
        if RUN_PARSER:
            print(f"  Parsed data: {os.path.abspath(PARSED_OUTPUT_DIR)}")
        if RUN_SPLITTER:
            print(f"  Split data: {os.path.abspath(SPLIT_OUTPUT_DIR)}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
