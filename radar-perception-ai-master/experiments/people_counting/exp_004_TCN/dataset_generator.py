#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

class RadarDataGenerator:
    def __init__(self, input_dir='split_data', output_dir='processed_dataset'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.features = ['x', 'y', 'z', 'velocity', 'snr']
        self.max_points = 1024
        self.frames_per_sample = 8
        self.frame_stride = 1

    def load_gt_map(self, summary_path):
        try:
            df = pd.read_csv(summary_path)
            if 'frame_num' not in df.columns or 'ground_truth' not in df.columns:
                return {}
            gt_map = {}
            for _, row in df.iterrows():
                frame = int(row['frame_num'])
                label = row['ground_truth']
                if isinstance(label, str):
                    digits = ''.join(c for c in label if c.isdigit())
                    gt_map[frame] = int(digits) if digits else 0
                else:
                    gt_map[frame] = int(label) if label else 0
            return gt_map
        except Exception as e:
            print(f"Error loading summary: {e}")
            return {}

    def find_summary_path(self, pc_path):
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

    def process_frame_group(self, df_combined):
        # Truncate to max_points using highest SNR
        if len(df_combined) > self.max_points and 'snr' in df_combined.columns:
            df_combined = df_combined.nlargest(self.max_points, 'snr')

        # Hierarchical sort: x, y, z
        df_combined = df_combined.sort_values(by=['x', 'y', 'z'], ascending=True)

        # Extract features (no normalization here)
        feature_data = []
        for feat in self.features:
            if feat in df_combined.columns:
                feature_data.append(df_combined[feat].values.astype(np.float32))

        if not feature_data:
            return None

        data = np.column_stack(feature_data)

        # Pad if needed
        current_n = data.shape[0]
        if current_n < self.max_points:
            pad_n = self.max_points - current_n
            padding = np.zeros((pad_n, len(self.features)), dtype=np.float32)
            data = np.vstack([data, padding])
        else:
            data = data[:self.max_points]

        # Output shape: (1024, 5) - ready for TCN
        return data

    def process_file(self, pc_path):
        print(f"Processing: {os.path.basename(pc_path)}")
        try:
            df_pc = pd.read_csv(pc_path)
        except Exception as e:
            print(f"Error loading file: {e}")
            return None, None

        if df_pc.empty or 'frame_num' not in df_pc.columns:
            print("Empty or invalid file")
            return None, None

        summary_path = self.find_summary_path(pc_path)
        gt_map = self.load_gt_map(summary_path) if summary_path else {}

        unique_frames = sorted(df_pc['frame_num'].unique())
        all_samples = []
        all_labels = []

        for start_i in range(0, len(unique_frames), self.frame_stride):
            end_i = start_i + self.frames_per_sample
            if end_i > len(unique_frames):
                break

            frame_window = unique_frames[start_i:end_i]
            combined_df = pd.concat([df_pc[df_pc['frame_num'] == f] for f in frame_window], ignore_index=True)

            if combined_df.empty:
                continue

            data = self.process_frame_group(combined_df)
            if data is None:
                continue

            all_samples.append(data)
            first_frame = frame_window[0]
            label = gt_map.get(first_frame, 0)
            all_labels.append(label)

        if not all_samples:
            return None, None

        samples_array = np.array(all_samples, dtype=np.float32)
        labels_array = np.array(all_labels, dtype=np.int32)

        print(f"Generated {len(all_samples)} samples")
        return samples_array, labels_array

    def collect_files(self):
        if not os.path.exists(self.input_dir):
            print(f"Input dir not found: {self.input_dir}")
            return {}

        splits = {'train': [], 'val': [], 'test': []}
        for fname in os.listdir(self.input_dir):
            if '_pointcloud' in fname and fname.endswith('.csv'):
                fpath = os.path.join(self.input_dir, fname)
                if '_train.csv' in fname:
                    splits['train'].append(fpath)
                elif '_val.csv' in fname:
                    splits['val'].append(fpath)
                elif '_test.csv' in fname:
                    splits['test'].append(fpath)
        return splits

    def process_split(self, files):
        all_data, all_labels = [], []
        for fpath in files:
            data, labels = self.process_file(fpath)
            if data is not None:
                all_data.append(data)
            if labels is not None:
                all_labels.append(labels)

        if not all_data:
            return None, None

        combined_data = np.concatenate(all_data, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        return combined_data, combined_labels

    def run(self):
        print("="*80)
        print("RADAR DATASET GENERATOR")
        print("="*80)
        print(f"Features: {self.features}")
        print(f"Max points: {self.max_points}")
        print(f"Output shape: (N, {self.max_points}, {len(self.features)})")
        print(f"Frames per sample: {self.frames_per_sample}")
        print("="*80)

        os.makedirs(self.output_dir, exist_ok=True)
        files_by_split = self.collect_files()

        for split in ['train', 'val', 'test']:
            flist = files_by_split.get(split, [])
            if not flist:
                continue

            print(f"\nProcessing {split}...")
            data, labels = self.process_split(flist)

            if data is None:
                print(f"No data for {split}")
                continue

            print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
            np.save(os.path.join(self.output_dir, f'data_{split}.npy'), data)
            np.save(os.path.join(self.output_dir, f'labels_{split}.npy'), labels)
            print(f"Saved {split} data")

        # Save normalization info for later use
        print("\nComputing normalization statistics from training data...")
        train_path = os.path.join(self.output_dir, 'data_train.npy')
        if os.path.exists(train_path):
            featuremap_train = np.load(train_path)
            # Data is already (batch, 1024, 5), no reshape needed

            train_valid_mask = np.any(featuremap_train != 0, axis=-1)
            train_valid_points = featuremap_train[train_valid_mask]
            train_mean = np.mean(train_valid_points, axis=0)
            train_std = np.std(train_valid_points, axis=0)
            train_std[train_std == 0] = 1.0

            norm_path = os.path.join(self.output_dir, 'NormData.txt')
            with open(norm_path, 'w') as f:
                f.write("--- Computing Statistics for Normalization ---\n")
                f.write(f"Feature Mean: {train_mean}\n")
                f.write(f"Feature Std:  {train_std}\n")
            print(f"Feature Mean: {train_mean}")
            print(f"Feature Std:  {train_std}")
            print(f"Normalization stats saved to {norm_path}")

        print("\nProcessing complete!")

if __name__ == "__main__":
    generator = RadarDataGenerator(input_dir='split_data', output_dir='processed_dataset')
    generator.run()
