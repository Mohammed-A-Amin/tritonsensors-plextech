#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import binned_statistic_2d
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# =========================
# CONFIGURATION
# =========================

@dataclass
class HeatmapConfig:
    """Configuration for heatmap image dataset generation."""
    
    # Input/Output paths
    input_dir: str = "parsed_data"
    output_dir: str = "heatmap_dataset"
    summary_csv_pattern: str = "_summary.csv"
    pointcloud_csv_pattern: str = "_pointcloud.csv"
    
    # Heatmap resolution
    resolution: int = 64  # Grid size (64x64, 128x128, etc.)
    
    # Image spatial limits (for visualization/binning)
    image_x_limits: Tuple[float, float] = (-6.0, 6.0)
    image_z_limits: Tuple[float, float] = (-6.0, 6.0)
    
    # Point cloud filtering (can differ from image limits)
    apply_bounding_box: bool = True  # Filter points outside bbox
    bbox_x_limits: Optional[Tuple[float, float]] = (-8.0, 8.0)  # None = use image limits
    bbox_z_limits: Optional[Tuple[float, float]] = (-8.0, 8.0)  # None = use image limits
    min_velocity: Optional[float] = 0.1  # Remove points below this velocity (None to disable)
    
    # Intensity/heatmap settings
    intensity_feature: str = "snr"  # 'snr', 'velocity', 'intensity'
    use_abs_velocity: bool = True  # Use absolute value when intensity_feature='velocity'
    bin_statistic: str = "mean"  # 'mean', 'sum', 'max', 'min', 'count', 'median'
    background_value: float = 0.0
    interpolation: str = "bilinear"  # 'bilinear', 'nearest', 'gaussian'
    
    # Frame aggregation (default OFF)
    aggregate_frames: bool = False
    frames_per_sample: int = 1  # Number of consecutive frames to aggregate
    frame_stride: int = 1  # Stride between aggregated samples
    aggregation_mode: str = "concat"  # 'concat', 'mean', 'sum', 'max'
    
    # Output naming
    prefix: str = "radar"
    suffix_mode: str = "label"  # 'label' (person count) or 'split' (train/val/test)
    
    # Output options
    save_images: bool = True
    save_numpy: bool = False  # Optional: save as .npy for direct loading
    create_groundtruth_csv: bool = True
    image_format: str = "png"  # 'png', 'jpg'
    
    # Image settings
    dpi: int = 100
    
    verbose: bool = True


# =========================
# HEATMAP GENERATOR CLASS
# =========================

class RadarHeatmapGenerator:
    """Generate fixed-resolution heatmap images from radar point clouds."""
    
    def __init__(self, config: HeatmapConfig):
        self.config = config
        
        # Validate aggregation mode
        valid_agg_modes = ['concat', 'mean', 'sum', 'max']
        if self.config.aggregation_mode not in valid_agg_modes:
            raise ValueError(f"aggregation_mode must be one of {valid_agg_modes}")
        
        # Validate bin statistic
        valid_bin_stats = ['mean', 'sum', 'max', 'min', 'count', 'median']
        if self.config.bin_statistic not in valid_bin_stats:
            raise ValueError(f"bin_statistic must be one of {valid_bin_stats}")
        
        # Create custom colormap (blue to yellow intensity)
        colors = ['#0000FF', '#0080FF', '#00FFFF', '#80FF00', '#FFFF00']
        self.cmap = LinearSegmentedColormap.from_list('radar_intensity', colors, N=256)
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Determine effective bounding box limits
        self.effective_bbox_x = self.config.bbox_x_limits if self.config.bbox_x_limits is not None else self.config.image_x_limits
        self.effective_bbox_z = self.config.bbox_z_limits if self.config.bbox_z_limits is not None else self.config.image_z_limits
        
        self._log(f"Initialized HeatmapGenerator")
        self._log(f"  Resolution: {self.config.resolution}x{self.config.resolution}")
        self._log(f"  Image X limits: {self.config.image_x_limits}")
        self._log(f"  Image Z limits: {self.config.image_z_limits}")
        if self.config.apply_bounding_box:
            self._log(f"  BBox X limits: {self.effective_bbox_x}")
            self._log(f"  BBox Z limits: {self.effective_bbox_z}")
        self._log(f"  Intensity feature: {self.config.intensity_feature}")
        self._log(f"  Bin statistic: {self.config.bin_statistic}")
        if self.config.intensity_feature == 'velocity':
            self._log(f"  Use abs(velocity): {self.config.use_abs_velocity}")
        self._log(f"  Save NumPy: {self.config.save_numpy}")
        self._log(f"  Aggregate frames: {self.config.aggregate_frames}")
        if self.config.aggregate_frames:
            self._log(f"  Aggregation mode: {self.config.aggregation_mode}")
            self._log(f"  Frames per sample: {self.config.frames_per_sample}")
            self._log(f"  Frame stride: {self.config.frame_stride}")
        
    def _log(self, msg: str):
        """Log message if verbose enabled."""
        if self.config.verbose:
            print(msg)
    
    def _filter_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply velocity and bounding box filters to point cloud."""
        if len(df) == 0:
            return df
        
        original_count = len(df)
        
        # Filter by velocity (using absolute value if configured)
        if self.config.min_velocity is not None and 'velocity' in df.columns:
            if self.config.use_abs_velocity:
                df = df[df['velocity'].abs() >= self.config.min_velocity].copy()
            else:
                df = df[df['velocity'] >= self.config.min_velocity].copy()
            if self.config.verbose and original_count > 0:
                self._log(f"    Velocity filter: {original_count} -> {len(df)} points")
        
        # Filter by bounding box (using separate bbox limits)
        if self.config.apply_bounding_box:
            x_min, x_max = self.effective_bbox_x
            z_min, z_max = self.effective_bbox_z
            
            mask = (
                (df['x'] >= x_min) & (df['x'] <= x_max) &
                (df['z'] >= z_min) & (df['z'] <= z_max)
            )
            df = df[mask].copy()
            if self.config.verbose and original_count > 0:
                self._log(f"    Bounding box filter: -> {len(df)} points")
        
        return df
    
    def _create_heatmap_array(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create fixed-resolution heatmap from point cloud data.
        Uses image_x_limits and image_z_limits for binning.
        Returns intensity grid.
        """
        resolution = self.config.resolution
        x_range = self.config.image_x_limits
        z_range = self.config.image_z_limits
        
        # Initialize background
        intensity_map = np.full((resolution, resolution), 
                               self.config.background_value, 
                               dtype=np.float32)
        
        if len(df) == 0:
            return intensity_map
        
        # Create bin edges (using image limits, not bbox limits)
        x_bins = np.linspace(x_range[0], x_range[1], resolution + 1)
        z_bins = np.linspace(z_range[0], z_range[1], resolution + 1)
        
        # Get intensity values
        if self.config.intensity_feature not in df.columns:
            intensity_values = np.ones(len(df))
        else:
            intensity_values = df[self.config.intensity_feature].values
            
            # Apply absolute value for velocity if configured
            if self.config.intensity_feature == 'velocity' and self.config.use_abs_velocity:
                intensity_values = np.abs(intensity_values)
        
        # Bin statistics with configurable statistic
        intensity_grid, x_edges, z_edges, bin_number = binned_statistic_2d(
            df['x'].values, 
            df['z'].values, 
            intensity_values,
            statistic=self.config.bin_statistic,
            bins=[x_bins, z_bins],
            expand_binnumbers=True
        )
        
        # Replace NaN with background value
        intensity_grid = np.nan_to_num(intensity_grid, nan=self.config.background_value)
        intensity_map = intensity_grid.T  # Transpose for proper orientation
        
        return intensity_map
    
    def _aggregate_heatmaps(self, heatmaps: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate multiple heatmaps based on aggregation mode.
        
        Args:
            heatmaps: List of heatmap arrays to aggregate
            
        Returns:
            Aggregated heatmap array
        """
        if len(heatmaps) == 0:
            return np.full((self.config.resolution, self.config.resolution), 
                          self.config.background_value, dtype=np.float32)
        
        if len(heatmaps) == 1:
            return heatmaps[0]
        
        stacked = np.stack(heatmaps, axis=0)
        
        if self.config.aggregation_mode == 'mean':
            return np.mean(stacked, axis=0)
        elif self.config.aggregation_mode == 'sum':
            return np.sum(stacked, axis=0)
        elif self.config.aggregation_mode == 'max':
            return np.max(stacked, axis=0)
        else:  # 'concat' - create from concatenated point clouds
            # This is handled differently - we concat points before creating heatmap
            return heatmaps[0]  # Should not reach here for concat mode
    
    def _save_heatmap_image(self, heatmap: np.ndarray, output_path: str, 
                           intensity_range: Tuple[float, float]):
        """Save heatmap as clean image file without axes or labels."""
        # Create figure with exact pixel dimensions
        fig_width = self.config.resolution / self.config.dpi
        fig_height = self.config.resolution / self.config.dpi
        
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=self.config.dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        vmin, vmax = intensity_range
        
        # Avoid division by zero
        if vmin == vmax:
            vmax = vmin + 1.0
        
        # Display heatmap without any axes/labels (using image limits)
        ax.imshow(
            heatmap,
            extent=[self.config.image_x_limits[0], self.config.image_x_limits[1],
                   self.config.image_z_limits[0], self.config.image_z_limits[1]],
            origin='lower',
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
            interpolation=self.config.interpolation
        )
        
        plt.savefig(output_path, dpi=self.config.dpi)
        plt.close(fig)
    
    def _aggregate_point_clouds(self, df: pd.DataFrame, frame_list: List[int]) -> pd.DataFrame:
        """Aggregate multiple frames into single point cloud (for concat mode)."""
        frames_data = [df[df['frame_num'] == f].copy() for f in frame_list]
        aggregated = pd.concat(frames_data, ignore_index=True)
        return aggregated
    
    def load_ground_truth_map(self, summary_path: str) -> Dict[int, int]:
        """Load frame number to ground truth label mapping."""
        try:
            df_summary = pd.read_csv(summary_path)
            if 'frame_num' not in df_summary.columns or 'ground_truth' not in df_summary.columns:
                self._log(f"  Warning: Required columns not found in {summary_path}")
                return {}
            
            gt_map = {}
            for _, row in df_summary.iterrows():
                frame = int(row['frame_num'])
                label = row['ground_truth']
                
                # Parse label if it's string format like "3_person"
                if isinstance(label, str):
                    digits = ''.join(c for c in label if c.isdigit())
                    gt_map[frame] = int(digits) if digits else 0
                else:
                    gt_map[frame] = int(label) if pd.notna(label) else 0
            
            return gt_map
        
        except Exception as e:
            self._log(f"  Error loading summary: {e}")
            return {}
    
    def process_file(self, pointcloud_path: str, summary_path: str) -> List[Dict]:
        """
        Process a single pointcloud file and generate heatmap dataset.
        Returns list of sample metadata.
        """
        self._log(f"\nProcessing: {os.path.basename(pointcloud_path)}")
        
        # Load data
        try:
            df_pc = pd.read_csv(pointcloud_path)
        except Exception as e:
            self._log(f"  Error loading pointcloud: {e}")
            return []
        
        if df_pc.empty or 'frame_num' not in df_pc.columns:
            self._log("  Empty or invalid pointcloud file")
            return []
        
        # Load ground truth
        gt_map = self.load_ground_truth_map(summary_path) if os.path.exists(summary_path) else {}
        
        # Get base filename for output naming
        base_name = os.path.basename(pointcloud_path).replace('_pointcloud.csv', '')
        base_name = base_name.replace('_train.csv', '').replace('_val.csv', '').replace('_test.csv', '')
        
        # Determine split from filename
        if '_train' in pointcloud_path:
            split = 'train'
        elif '_val' in pointcloud_path:
            split = 'val'
        elif '_test' in pointcloud_path:
            split = 'test'
        else:
            split = 'all'
        
        unique_frames = sorted(df_pc['frame_num'].unique())
        self._log(f"  Found {len(unique_frames)} frames")
        
        samples_metadata = []
        
        # Process frames based on aggregation mode
        if self.config.aggregate_frames and self.config.frames_per_sample > 1:
            # ===== AGGREGATED MODE =====
            self._log(f"  Using aggregation mode: {self.config.aggregation_mode}")
            
            # Calculate number of samples
            num_samples = (len(unique_frames) - self.config.frames_per_sample) // self.config.frame_stride + 1
            self._log(f"  Will generate approximately {num_samples} aggregated samples")
            
            sample_idx = 0
            for start_idx in range(0, len(unique_frames) - self.config.frames_per_sample + 1, self.config.frame_stride):
                frame_window = unique_frames[start_idx:start_idx + self.config.frames_per_sample]
                
                if len(frame_window) < self.config.frames_per_sample:
                    break  # Not enough frames left
                
                first_frame = frame_window[0]
                last_frame = frame_window[-1]
                
                # Get label (from first frame)
                label = gt_map.get(first_frame, 0)
                
                # Generate heatmap based on aggregation mode
                if self.config.aggregation_mode == 'concat':
                    # Concatenate point clouds first, then create single heatmap
                    df_aggregated = self._aggregate_point_clouds(df_pc, frame_window)
                    df_filtered = self._filter_points(df_aggregated)
                    
                    if len(df_filtered) == 0:
                        continue
                    
                    heatmap = self._create_heatmap_array(df_filtered)
                    
                    # Determine intensity range
                    if self.config.intensity_feature in df_filtered.columns:
                        intensity_vals = df_filtered[self.config.intensity_feature].values
                        if self.config.intensity_feature == 'velocity' and self.config.use_abs_velocity:
                            intensity_vals = np.abs(intensity_vals)
                        intensity_range = (intensity_vals.min(), intensity_vals.max())
                    else:
                        intensity_range = (0, 1)
                
                else:
                    # mean, sum, max: create heatmaps for each frame, then aggregate
                    heatmaps = []
                    all_intensities = []
                    
                    for frame_num in frame_window:
                        df_frame = df_pc[df_pc['frame_num'] == frame_num].copy()
                        df_filtered = self._filter_points(df_frame)
                        
                        frame_heatmap = self._create_heatmap_array(df_filtered)
                        heatmaps.append(frame_heatmap)
                        
                        # Collect intensities for range calculation
                        if len(df_filtered) > 0 and self.config.intensity_feature in df_filtered.columns:
                            intensity_vals = df_filtered[self.config.intensity_feature].values
                            if self.config.intensity_feature == 'velocity' and self.config.use_abs_velocity:
                                intensity_vals = np.abs(intensity_vals)
                            all_intensities.extend(intensity_vals)
                    
                    if len(heatmaps) == 0:
                        continue
                    
                    # Aggregate heatmaps
                    heatmap = self._aggregate_heatmaps(heatmaps)
                    
                    # Determine intensity range
                    if len(all_intensities) > 0:
                        intensity_range = (min(all_intensities), max(all_intensities))
                    else:
                        intensity_range = (heatmap.min(), heatmap.max())
                
                # Generate output filename
                suffix = f"{label}person" if self.config.suffix_mode == "label" else split
                # filename = f"{self.config.prefix}_{base_name}_f{first_frame:04d}-{last_frame:04d}_{suffix}"
                filename = f"{self.config.prefix}_f{first_frame:04d}-{last_frame:04d}_{suffix}"
                
                # Save outputs
                if self.config.save_images:
                    img_path = os.path.join(self.config.output_dir, 
                                           f"{filename}.{self.config.image_format}")
                    self._save_heatmap_image(heatmap, img_path, intensity_range)
                
                if self.config.save_numpy:
                    npy_path = os.path.join(self.config.output_dir, f"{filename}.npy")
                    np.save(npy_path, heatmap)
                
                # Store metadata
                samples_metadata.append({
                    'filename': filename,
                    'frames': ','.join(map(str, frame_window)),
                    'first_frame': first_frame,
                    'last_frame': last_frame,
                    'num_frames': len(frame_window),
                    'label': label,
                    'split': split,
                    'base_name': base_name,
                    'aggregation_mode': self.config.aggregation_mode
                })
                
                sample_idx += 1
        
        else:
            # ===== SINGLE FRAME MODE (default) =====
            for frame_num in unique_frames:
                df_frame = df_pc[df_pc['frame_num'] == frame_num].copy()
                df_filtered = self._filter_points(df_frame)
                
                if len(df_filtered) == 0:
                    continue
                
                # Create heatmap
                heatmap = self._create_heatmap_array(df_filtered)
                
                # Get label
                label = gt_map.get(frame_num, 0)
                
                # Generate output filename
                suffix = f"{label}person" if self.config.suffix_mode == "label" else split
                # filename = f"{self.config.prefix}_{base_name}_f{frame_num:04d}_{suffix}"
                filename = f"{self.config.prefix}_f{frame_num:04d}_{suffix}"
                
                # Determine intensity range
                if self.config.intensity_feature in df_filtered.columns:
                    intensity_vals = df_filtered[self.config.intensity_feature].values
                    if self.config.intensity_feature == 'velocity' and self.config.use_abs_velocity:
                        intensity_vals = np.abs(intensity_vals)
                    intensity_range = (intensity_vals.min(), intensity_vals.max())
                else:
                    intensity_range = (0, 1)
                
                # Save outputs
                if self.config.save_images:
                    img_path = os.path.join(self.config.output_dir, 
                                           f"{filename}.{self.config.image_format}")
                    self._save_heatmap_image(heatmap, img_path, intensity_range)
                
                if self.config.save_numpy:
                    npy_path = os.path.join(self.config.output_dir, f"{filename}.npy")
                    np.save(npy_path, heatmap)
                
                # Store metadata
                samples_metadata.append({
                    'filename': filename,
                    'frame_num': frame_num,
                    'label': label,
                    'split': split,
                    'base_name': base_name
                })
        
        self._log(f"  Generated {len(samples_metadata)} heatmap samples")
        return samples_metadata
    
    def run(self):
        """Main pipeline to process all files."""
        self._log(f"\n{'='*60}")
        self._log(f"RADAR HEATMAP DATASET GENERATOR")
        self._log(f"{'='*60}")
        
        # Find all pointcloud files
        if not os.path.exists(self.config.input_dir):
            self._log(f"Error: Input directory not found: {self.config.input_dir}")
            return
        
        all_files = os.listdir(self.config.input_dir)
        pointcloud_files = [f for f in all_files if self.config.pointcloud_csv_pattern in f]
        
        self._log(f"\nFound {len(pointcloud_files)} pointcloud files")
        
        all_metadata = []
        
        for pc_file in pointcloud_files:
            pc_path = os.path.join(self.config.input_dir, pc_file)
            
            # Find corresponding summary file
            base = pc_file.replace(self.config.pointcloud_csv_pattern, '')
            base = base.replace('_train', '').replace('_val', '').replace('_test', '')
            summary_file = f"{base}{self.config.summary_csv_pattern}"
            
            # Try multiple summary file patterns
            summary_paths = [
                os.path.join(self.config.input_dir, summary_file),
                os.path.join(self.config.input_dir, summary_file.replace('.csv', '_all.csv')),
                os.path.join(self.config.input_dir, f"{base}_summary_all.csv")
            ]
            
            summary_path = None
            for sp in summary_paths:
                if os.path.exists(sp):
                    summary_path = sp
                    break
            
            if summary_path is None:
                self._log(f"  Warning: No summary file found for {pc_file}")
                summary_path = ""  # Will be handled gracefully
            
            # Process file
            metadata = self.process_file(pc_path, summary_path)
            all_metadata.extend(metadata)
        
        # Save ground truth CSV
        if self.config.create_groundtruth_csv and all_metadata:
            df_metadata = pd.DataFrame(all_metadata)
            csv_path = os.path.join(self.config.output_dir, 'groundtruth.csv')
            df_metadata.to_csv(csv_path, index=False)
            self._log(f"\nSaved ground truth CSV: {csv_path}")
            
            # Print statistics
            self._log(f"\n{'='*60}")
            self._log(f"GENERATION COMPLETE")
            self._log(f"{'='*60}")
            self._log(f"Total samples: {len(all_metadata)}")
            
            if 'split' in df_metadata.columns:
                split_counts = df_metadata['split'].value_counts()
                for split, count in split_counts.items():
                    self._log(f"  {split}: {count} samples")
            
            if 'label' in df_metadata.columns:
                label_counts = df_metadata['label'].value_counts().sort_index()
                self._log(f"\nLabel distribution:")
                for label, count in label_counts.items():
                    self._log(f"  {label} person(s): {count} samples")
            
            self._log(f"\nOutput directory: {os.path.abspath(self.config.output_dir)}")
            self._log(f"{'='*60}\n")


# =========================
# MAIN EXECUTION
# =========================

def main():
    """Main configuration and execution."""
    
    # ====================
    # EDIT CONFIGURATION HERE
    # ====================
    
    config = HeatmapConfig(
        # Input/Output
        input_dir="parsed_data_sample",
        output_dir="heatmap_dataset",
        
        # Image settings
        resolution=32,  # 64x64, 128x128, etc.
        
        # Image spatial limits (for visualization/binning)
        image_x_limits=(-6.0, 6.0),
        image_z_limits=(-6.0, 6.0),
        
        # Bounding box filtering (can differ from image limits)
        apply_bounding_box=True,
        bbox_x_limits=(-3.0, 3.0),  # Filter points outside this range (None = use image_x_limits)
        bbox_z_limits=(-3.0, 3.0),  # Filter points outside this range (None = use image_z_limits)
        
        # Velocity filtering
        min_velocity=0.0,  # Filter static points (None to disable)
        
        # Intensity settings
        intensity_feature="snr",  # 'snr', 'velocity', 'intensity'
        use_abs_velocity=True,  # Use abs() when intensity_feature='velocity'
        bin_statistic="mean",  # 'mean', 'sum', 'max', 'min', 'count', 'median'
        
        # Frame aggregation (default OFF)
        aggregate_frames=False,  # Set True to enable
        frames_per_sample=8,  # Number of frames to aggregate
        frame_stride=1,  # Stride between samples
        aggregation_mode="concat",  # 'concat', 'mean', 'sum', 'max'
        
        # Output options
        prefix="radar",
        suffix_mode="label",  # 'label' or 'split'
        save_images=True,
        save_numpy=False,  # Set True to also save .npy files
        image_format="png",
        
        # Other
        verbose=True
    )
    
    # Run generator
    generator = RadarHeatmapGenerator(config)
    generator.run()


if __name__ == "__main__":
    main()
