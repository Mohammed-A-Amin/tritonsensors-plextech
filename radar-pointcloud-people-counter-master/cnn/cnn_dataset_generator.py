#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration for radar dataset generation."""
    # Paths
    input_dir: str = "split_data"
    output_dir: str = "processed_dataset"
    
    # Features (enabled by default)
    features: List[str] = None  # e.g., ['x', 'y', 'z', 'velocity', 'snr']
    
    # Points and padding
    max_points: int = 64
    use_zero_pad: bool = True
    pad_value: float = 0.0
    truncate_method: str = "highest_snr"  # 'highest_snr' or 'sort_based'
    
    # Hierarchical sorting (applied after truncation)
    sort_by: List[str] = None  # e.g., ['x', 'y', 'z'] for position hierarchy
    sort_descending: bool = False  # Ascending for positions
    
    # Per-feature normalization
    normalize_these: List[str] = None  # Features to normalize
    norm_method: str = "min_max"
    feature_ranges: Dict[str, Tuple[float, float]] = None
    
    # Reshaping to channels
    use_reshape: bool = False
    reshape_h: int = 8
    reshape_w: int = 8
    
    # Temporal (concat frames before processing)
    frames_per_sample: int = 1
    frame_stride: int = 1
    
    # Labels
    label_type: str = "count"
    
    # Options
    verbose: bool = True
    save_separate_splits: bool = True


# =========================
# CONFIGURATION PRESETS
# =========================

def preset_basic() -> Config:
    """
    Basic configuration: 5 features, 64 points, no reshape
    Output shape: (N, 64, 5)
    """
    features = ['x', 'y', 'z', 'velocity', 'snr']
    ranges = {
        'x': (-3.0, 3.0),
        'y': (0.0, 2.7),
        'z': (-3.0, 3.0),
        'velocity': (-2.0, 2.0),
        'snr': (0.0, 100.0)
    }
    return Config(
        features=features,
        feature_ranges=ranges,
        sort_by=['x', 'y', 'z'],
        normalize_these=['velocity', 'snr'],
        max_points=64,
        use_zero_pad=True,
        use_reshape=False
    )


# =========================
# CONFIGURATION BUILDER
# =========================

def build_custom_config() -> Config:
    """
    Interactive configuration builder.
    Returns a Config object based on user choices.
    """
    print("\n" + "="*60)
    print("RADAR DATASET CONFIGURATION BUILDER")
    print("="*60)
    
    # Feature selection
    print("\n[1] SELECT FEATURES")
    print("Available features: x, y, z, velocity, snr, noise, rcs")
    print("Default: x, y, z, velocity, snr")
    features_input = input("Enter features (comma-separated) or press Enter for default: ").strip()
    if features_input:
        features = [f.strip() for f in features_input.split(',')]
    else:
        features = ['x', 'y', 'z', 'velocity', 'snr']
    print(f"✓ Selected features: {features}")
    
    # Max points
    print("\n[2] SET MAX POINTS")
    print("Common values: 64 (8x8), 121 (11x11), 128, 256 (16x16)")
    max_points_input = input("Enter max points (default 64): ").strip()
    max_points = int(max_points_input) if max_points_input else 64
    print(f"✓ Max points: {max_points}")
    
    # Zero padding
    print("\n[3] ZERO PADDING")
    use_pad = input("Enable zero-padding? (y/n, default y): ").strip().lower()
    use_zero_pad = use_pad != 'n'
    print(f"✓ Zero-padding: {'Enabled' if use_zero_pad else 'Disabled'}")
    
    # Normalization
    print("\n[4] NORMALIZATION")
    print("Which features to normalize?")
    print("Options: all, none, velocity+snr (default), positions (x,y,z), custom")
    norm_choice = input("Choice (default: velocity+snr): ").strip().lower()
    if norm_choice == 'all':
        normalize_these = features.copy()
    elif norm_choice == 'none':
        normalize_these = []
    elif norm_choice == 'positions':
        normalize_these = [f for f in features if f in ['x', 'y', 'z']]
    elif norm_choice == 'custom':
        norm_input = input("Enter features to normalize (comma-separated): ").strip()
        normalize_these = [f.strip() for f in norm_input.split(',')]
    else:  # Default: velocity+snr
        normalize_these = [f for f in features if f in ['velocity', 'snr']]
    print(f"✓ Normalize: {normalize_these}")
    
    # Feature ranges
    feature_ranges = {
        'x': (-3.0, 3.0),
        'y': (0.0, 2.7),
        'z': (-3.0, 3.0),
        'velocity': (-2.0, 2.0),
        'snr': (0.0, 100.0),
        'noise': (0.0, 50.0),
        'rcs': (-10.0, 10.0)
    }
    
    # Reshape
    print("\n[5] RESHAPE TO CHANNELS")
    use_reshape_input = input("Enable reshape to HxWxC? (y/n, default n): ").strip().lower()
    use_reshape = use_reshape_input == 'y'
    reshape_h, reshape_w = 8, 8
    if use_reshape:
        print(f"For {max_points} points, suggest:")
        if max_points == 64:
            print("  8x8")
        elif max_points == 121:
            print("  11x11")
        elif max_points == 128:
            print("  16x8 or 8x16")
        elif max_points == 256:
            print("  16x16")
        reshape_input = input(f"Enter HxW (e.g., 8x8): ").strip()
        if 'x' in reshape_input:
            h_str, w_str = reshape_input.split('x')
            reshape_h, reshape_w = int(h_str), int(w_str)
        print(f"✓ Reshape: {reshape_h}x{reshape_w}x{len(features)}")
    else:
        print("✓ Reshape: Disabled")
    
    # Temporal
    print("\n[6] TEMPORAL FRAMES")
    frames_input = input("Frames per sample (default 1): ").strip()
    frames_per_sample = int(frames_input) if frames_input else 1
    print(f"✓ Frames per sample: {frames_per_sample}")
    
    # Truncation
    print("\n[7] TRUNCATION METHOD")
    print("Options: highest_snr (default), sort_based")
    trunc_input = input("Method: ").strip().lower()
    truncate_method = trunc_input if trunc_input in ['highest_snr', 'sort_based'] else 'highest_snr'
    print(f"✓ Truncation: {truncate_method}")
    
    print("\n" + "="*60)
    print("Configuration complete!")
    print("="*60 + "\n")
    
    return Config(
        features=features,
        feature_ranges=feature_ranges,
        sort_by=['x', 'y', 'z'],
        normalize_these=normalize_these,
        max_points=max_points,
        use_zero_pad=use_zero_pad,
        use_reshape=use_reshape,
        reshape_h=reshape_h,
        reshape_w=reshape_w,
        frames_per_sample=frames_per_sample,
        truncate_method=truncate_method
    )


def print_config_summary(config: Config):
    """Print a readable summary of the configuration."""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Features ({len(config.features)}): {config.features}")
    print(f"Max points: {config.max_points}")
    print(f"Zero-padding: {'Enabled' if config.use_zero_pad else 'Disabled'}")
    print(f"Sort by: {config.sort_by} (hierarchical)")
    print(f"Normalize: {config.normalize_these if config.normalize_these else 'None'}")
    if config.use_reshape:
        print(f"Reshape: {config.reshape_h}x{config.reshape_w}x{len(config.features)}")
        print(f"Output shape: (N, {config.reshape_h}, {config.reshape_w}, {len(config.features)})")
    else:
        print(f"Output shape: (N, {config.max_points}, {len(config.features)})")
    print(f"Frames per sample: {config.frames_per_sample}")
    print(f"Truncation: {config.truncate_method}")
    print("="*60 + "\n")


# =========================
# MAIN GENERATOR CLASS
# =========================

class UpdatedRadarGenerator:
    """Radar dataset processor with configuration helpers."""
    
    def __init__(self, config: Config):
        self.config = config
        if self.config.features is None:
            self.config.features = ['x', 'y', 'z', 'velocity', 'snr']
        if self.config.normalize_these is None:
            self.config.normalize_these = ['velocity', 'snr']
        if self.config.sort_by is None:
            self.config.sort_by = ['x', 'y', 'z']
        self.num_features = len(self.config.features)
        self._validate()
    
    def _validate(self):
        """Validation with helpful error messages."""
        if self.num_features == 0:
            raise ValueError("ERROR: At least one feature required. Check config.features.")
        
        if self.config.use_reshape:
            expected = self.config.reshape_h * self.config.reshape_w
            if self.config.max_points != expected:
                raise ValueError(
                    f"ERROR: Reshape mismatch!\n"
                    f"  reshape_h={self.config.reshape_h}, reshape_w={self.config.reshape_w}\n"
                    f"  Expected max_points={expected}, but got {self.config.max_points}\n"
                    f"  Set max_points = {expected} or adjust reshape dimensions."
                )
        
        missing_sort = [col for col in self.config.sort_by if col not in self.config.features]
        if missing_sort:
            raise ValueError(
                f"ERROR: Sort columns {missing_sort} not in features {self.config.features}\n"
                f"  Remove from sort_by or add to features."
            )
        
        missing_norm = [f for f in self.config.normalize_these if f not in self.config.features]
        if missing_norm:
            print(f"WARNING: Normalize features {missing_norm} not in selected features. Ignored.")
    
    def _log(self, msg: str):
        """Log if verbose."""
        if self.config.verbose:
            print(msg)
    
    def _normalize_feature(self, values: np.ndarray, feature_name: str) -> np.ndarray:
        """Min-max normalize only if feature in normalize_these."""
        if feature_name not in self.config.normalize_these:
            return values
        if feature_name not in self.config.feature_ranges:
            self._log(f"Warning: No range for {feature_name}, skipping norm.")
            return values
        min_val, max_val = self.config.feature_ranges[feature_name]
        clipped = np.clip(values, min_val, max_val)
        range_val = max_val - min_val
        if range_val > 0:
            return (clipped - min_val) / range_val
        return np.zeros_like(values)
    
    def _truncate_by_snr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Truncate to max_points using highest SNR first."""
        if len(df) <= self.config.max_points or 'snr' not in df.columns:
            return df
        df = df.nlargest(self.config.max_points, 'snr')
        return df
    
    def _sort_hierarchical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hierarchical sort: x asc, then y asc, then z asc."""
        df = df.sort_values(by=self.config.sort_by, ascending=[True] * len(self.config.sort_by))
        return df
    
    def _sort_and_truncate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: truncate SNR first, then hierarchical sort."""
        if self.config.truncate_method == "highest_snr":
            df = self._truncate_by_snr(df)
        else:
            df = self._sort_hierarchical(df)
            if len(df) > self.config.max_points:
                df = df.head(self.config.max_points)
        
        df = self._sort_hierarchical(df)
        
        if len(df) > self.config.max_points:
            df = df.head(self.config.max_points)
        return df
    
    def _pad_points(self, frame_data: np.ndarray) -> np.ndarray:
        """Zero-pad if < max_points."""
        if not self.config.use_zero_pad:
            return frame_data
        current_n = frame_data.shape[0]
        if current_n >= self.config.max_points:
            return frame_data
        pad_n = self.config.max_points - current_n
        pad_shape = (pad_n, self.num_features)
        padding = np.full(pad_shape, self.config.pad_value, dtype=np.float32)
        return np.vstack([frame_data, padding])
    
    def _reshape_data(self, data: np.ndarray) -> np.ndarray:
        """Reshape to HxWxC if enabled."""
        if not self.config.use_reshape:
            return data
        try:
            return data.reshape(self.config.reshape_h, self.config.reshape_w, self.num_features)
        except ValueError as e:
            self._log(f"Warning: Reshape failed: {e}")
            return data
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and selectively normalize features."""
        feature_data = []
        for feat in self.config.features:
            if feat not in df.columns:
                self._log(f"Warning: Missing column {feat}.")
                continue
            col_data = df[feat].values.astype(np.float32)
            col_data = self._normalize_feature(col_data, feat)
            feature_data.append(col_data)
        if not feature_data:
            return np.empty((0, 0), dtype=np.float32)
        return np.column_stack(feature_data)
    
    def load_gt_map(self, summary_path: str) -> Dict[int, int]:
        """Load frame_num to ground_truth map."""
        try:
            df = pd.read_csv(summary_path)
            if 'frame_num' not in df.columns or 'ground_truth' not in df.columns:
                return {}
            gt_map = {}
            for _, row in df.iterrows():
                frame = int(row['frame_num'])
                label = row['ground_truth']
                if isinstance(label, str) and self.config.label_type == "count":
                    digits = ''.join(c for c in label if c.isdigit())
                    gt_map[frame] = int(digits) if digits else 0
                else:
                    gt_map[frame] = int(label) if label else 0
            return gt_map
        except Exception as e:
            self._log(f"Error loading summary {summary_path}: {e}")
            return {}
    
    def find_summary_path(self, pc_path: str) -> Optional[str]:
        """Find matching summary file."""
        base_dir = os.path.dirname(pc_path)
        filename = os.path.basename(pc_path)
        base_name = filename.replace('_pointcloud_train.csv', '').replace('_pointcloud_val.csv', '') \
                            .replace('_pointcloud_test.csv', '').replace('_pointcloud_all.csv', '')
        summary_path = os.path.join(base_dir, f"{base_name}_summary_all.csv")
        if not os.path.exists(summary_path):
            summary_path = os.path.join(base_dir, f"{base_name}_summary.csv")
        if not os.path.exists(summary_path):
            return None
        return summary_path
    
    def process_file(self, pc_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Process one pointcloud file."""
        self._log(f"Loading: {os.path.basename(pc_path)}")
        try:
            df_pc = pd.read_csv(pc_path)
        except Exception as e:
            self._log(f"Error loading PC: {e}")
            return None, None, None
        
        if df_pc.empty or 'frame_num' not in df_pc.columns:
            self._log("Empty or invalid PC file.")
            return None, None, None
        
        summary_path = self.find_summary_path(pc_path)
        gt_map = self.load_gt_map(summary_path) if summary_path else {}
        
        unique_frames = sorted(df_pc['frame_num'].unique())
        all_samples = []
        all_labels = []
        all_frame_ids = []
        
        for start_i in range(0, len(unique_frames), self.config.frame_stride):
            end_i = start_i + self.config.frames_per_sample
            if end_i > len(unique_frames):
                break
            frame_window = unique_frames[start_i:end_i]
            
            combined_df = pd.concat([df_pc[df_pc['frame_num'] == f] for f in frame_window], ignore_index=True)
            if combined_df.empty:
                continue
            
            combined_df = self._sort_and_truncate(combined_df)
            data = self._extract_features(combined_df)
            if data.shape[0] == 0:
                continue
            
            data = self._pad_points(data)
            data = self._reshape_data(data)
            
            all_samples.append(data)
            first_frame = frame_window[0]
            label = gt_map.get(first_frame, 0)
            all_labels.append(label)
            all_frame_ids.append(first_frame)
        
        if not all_samples:
            return None, None, None
        
        samples_array = np.array(all_samples, dtype=np.float32)
        labels_array = np.array(all_labels, dtype=np.int32)
        frames_array = np.array(all_frame_ids, dtype=np.int32)
        
        self._log(f"Processed {len(all_samples)} samples from {len(unique_frames)} frames.")
        return samples_array, labels_array, frames_array
    
    def collect_files(self) -> Dict[str, List[str]]:
        """Group files by split."""
        if not os.path.exists(self.config.input_dir):
            self._log(f"Input dir not found: {self.config.input_dir}")
            return {}
        splits = {'train': [], 'val': [], 'test': []}
        for fname in os.listdir(self.config.input_dir):
            if '_pointcloud' in fname and fname.endswith('.csv'):
                fpath = os.path.join(self.config.input_dir, fname)
                if '_train.csv' in fname:
                    splits['train'].append(fpath)
                elif '_val.csv' in fname:
                    splits['val'].append(fpath)
                elif '_test.csv' in fname:
                    splits['test'].append(fpath)
        return splits
    
    def process_split(self, files: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Process files for one split."""
        all_data, all_labels = [], []
        for fpath in files:
            data, labels, _ = self.process_file(fpath)
            if data is not None:
                all_data.append(data)
                if labels is not None:
                    all_labels.append(labels)
        if not all_data:
            return None, None
        combined_data = np.concatenate(all_data, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        return combined_data, combined_labels
    
    def save_config(self):
        """Save config as text."""
        cfg_path = os.path.join(self.config.output_dir, 'config.txt')
        with open(cfg_path, 'w') as f:
            f.write("Radar Dataset Configuration:\n")
            f.write(f"Features: {self.config.features} (C={self.num_features})\n")
            f.write(f"Max points: {self.config.max_points}\n")
            f.write(f"Zero-padding: {'Enabled' if self.config.use_zero_pad else 'Disabled'}\n")
            f.write(f"Sort by: {self.config.sort_by}\n")
            f.write(f"Normalize: {self.config.normalize_these}\n")
            if self.config.use_reshape:
                f.write(f"Reshape: {self.config.reshape_h}x{self.config.reshape_w}x{self.num_features}\n")
            f.write(f"Frames per sample: {self.config.frames_per_sample}\n")
            f.write(f"Truncation: {self.config.truncate_method}\n")
        self._log(f"Config saved: {cfg_path}")
    
    def run(self):
        """Main pipeline."""
        print_config_summary(self.config)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        files_by_split = self.collect_files()
        for split, flist in files_by_split.items():
            if flist:
                self._log(f"{split.capitalize()}: {len(flist)} files")
        
        for split in ['train', 'val', 'test']:
            flist = files_by_split.get(split, [])
            if not flist:
                continue
            self._log(f"\nProcessing {split}...")
            data, labels = self.process_split(flist)
            if data is None:
                self._log(f"No data for {split}")
                continue
            self._log(f"Data shape: {data.shape}, Labels: {labels.shape if labels is not None else 'None'}")
            if self.config.save_separate_splits:
                np.save(os.path.join(self.config.output_dir, f'data_{split}.npy'), data)
                if labels is not None:
                    np.save(os.path.join(self.config.output_dir, f'labels_{split}.npy'), labels)
                self._log(f"Saved {split} data/labels")
        
        self.save_config()
        self._log("\nProcessing complete!")


# =========================
# EXAMPLE USAGE
# =========================

def main():

    config = preset_basic()
    config.features = ['x', 'y', 'z', 'velocity', 'snr'] # Change features
    config.normalize_these = []
    config.max_points = 64
    config.use_zero_pad= True
    config.use_reshape = True
    config.reshape_h = 8
    config.reshape_w = 8
    config.frames_per_sample=3
    config.frame_stride=1        

    # Run generator
    generator = UpdatedRadarGenerator(config)
    generator.run()

if __name__ == "__main__":
    main()
