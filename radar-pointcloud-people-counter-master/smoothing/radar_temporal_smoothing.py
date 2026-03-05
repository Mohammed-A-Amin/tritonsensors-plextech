#!/usr/bin/env python3
"""
Radar Temporal Smoothing Analysis

This tool applies temporal smoothing methods to radar point cloud data
for people counting. It processes JSON radar data files and evaluates
smoothing performance using various metrics.
"""

from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import deque, Counter
import statistics
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import warnings
warnings.filterwarnings('ignore')


class TemporalSmoother:
    """Base class for temporal smoothing methods"""

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        """Reset internal state"""
        pass

    def smooth(self, count: int) -> int:
        """Process one frame count and return smoothed count"""
        raise NotImplementedError


class ExponentialSmoother(TemporalSmoother):
    """Exponential Smoothing"""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        super().__init__(f"ES(α={alpha})")

    def reset(self):
        self.smoothed_count = None

    def smooth(self, count: int) -> int:
        if self.smoothed_count is None:
            self.smoothed_count = float(count)
        else:
            self.smoothed_count = self.alpha * count + (1 - self.alpha) * self.smoothed_count
        return round(self.smoothed_count)


class MedianFilter(TemporalSmoother):
    """Median Filter"""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        super().__init__(f"Median(w={window_size})")

    def reset(self):
        self.window = deque(maxlen=self.window_size)

    def smooth(self, count: int) -> int:
        self.window.append(count)
        if len(self.window) >= 3:
            return int(statistics.median(self.window))
        return count


class MajorityVoting(TemporalSmoother):
    """Majority Voting with Tolerance"""

    def __init__(self, window_size: int = 7, tolerance: int = 1):
        self.window_size = window_size
        self.tolerance = tolerance
        super().__init__(f"Vote(w={window_size},t={tolerance})")

    def reset(self):
        self.window = deque(maxlen=self.window_size)

    def smooth(self, count: int) -> int:
        self.window.append(count)
        if len(self.window) < 3:
            return count

        counts = list(self.window)
        groups = {}
        for c in counts:
            found = False
            for key in groups:
                if abs(c - key) <= self.tolerance:
                    groups[key].append(c)
                    found = True
                    break
            if not found:
                groups[c] = [c]

        max_group = max(groups.items(), key=lambda x: len(x[1]))
        return round(sum(max_group[1]) / len(max_group[1]))


class CombinedSmoother(TemporalSmoother):
    """Combined: Median Filter + Exponential Smoothing"""

    def __init__(self, median_window: int = 5, alpha: float = 0.3):
        self.median_window = median_window
        self.alpha = alpha
        super().__init__(f"Combined(w={median_window},α={alpha})")

    def reset(self):
        self.window = deque(maxlen=self.median_window)
        self.smoothed_count = None

    def smooth(self, count: int) -> int:
        self.window.append(count)
        if len(self.window) >= 3:
            median_count = statistics.median(self.window)
        else:
            median_count = count

        if self.smoothed_count is None:
            self.smoothed_count = float(median_count)
        else:
            self.smoothed_count = self.alpha * median_count + (1 - self.alpha) * self.smoothed_count
        return round(self.smoothed_count)


class DelayedAggregation(TemporalSmoother):
    """Delayed 1-Second Output (Mode/Median)"""

    def __init__(self, frames_per_second: int = 4, method: str = "mode"):
        self.frames_per_output = frames_per_second
        self.method = method
        super().__init__(f"Delayed({method},fps={frames_per_second})")

    def reset(self):
        self.buffer = []
        self.frame_count = 0
        self.last_output = 0

    def smooth(self, count: int) -> int:
        self.buffer.append(count)
        self.frame_count += 1

        if self.frame_count >= self.frames_per_output:
            if self.method == "mode":
                count_freq = Counter(self.buffer)
                self.last_output = count_freq.most_common(1)[0][0]
            else:
                self.last_output = int(statistics.median(self.buffer))
            self.buffer = []
            self.frame_count = 0

        return self.last_output


def parse_people_label(lbl: str) -> int:
    """Parse labels like '1_person', '2_person', '3_person' to integer counts."""
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


def load_radar_json(path: str) -> Dict[str, Any]:
    """Load radar JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "data" not in data:
        raise ValueError(f"Invalid JSON structure in {path}: missing top-level 'data'")
    return data


def extract_track_counts(frames: List[Dict[str, Any]]) -> List[int]:
    """Extract per-frame track counts"""
    tracks_per_frame = []
    for fr in frames:
        fd = fr.get("frameData", {})
        if "numDetectedTracks" in fd and isinstance(fd["numDetectedTracks"], (int, float)):
            tracks_per_frame.append(int(fd["numDetectedTracks"]))
        else:
            trk = fd.get("trackData", [])
            tracks_per_frame.append(len(trk) if trk else 0)
    return tracks_per_frame


def compute_metrics(pred_counts: List[int], true_counts: List[int]) -> Dict[str, Any]:
    """Compute all metrics"""
    total = len(pred_counts)
    if total == 0:
        return dict(gfr=0.0, gfr1=0.0, gfr2=0.0, f1_weighted=0.0, f1_macro=0.0,
                   mdr=0.0, fdr=0.0, accuracy=0.0, mae=0.0)

    exact = sum(int(p == t) for p, t in zip(pred_counts, true_counts))
    within1 = sum(int(abs(p - t) <= 1) for p, t in zip(pred_counts, true_counts))
    within2 = sum(int(abs(p - t) <= 2) for p, t in zip(pred_counts, true_counts))
    miss = sum(int(p < t) for p, t in zip(pred_counts, true_counts))
    false = sum(int(p > t) for p, t in zip(pred_counts, true_counts))
    mae = np.mean([abs(p - t) for p, t in zip(pred_counts, true_counts)])

    labels_full = sorted(set(pred_counts + true_counts))
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_counts, pred_counts, labels=labels_full, average='weighted', zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        true_counts, pred_counts, labels=labels_full, average='macro', zero_division=0
    )

    return dict(
        gfr=100.0 * exact / total,
        gfr1=100.0 * within1 / total,
        gfr2=100.0 * within2 / total,
        f1_weighted=f1 * 100.0,
        f1_macro=f1_macro * 100.0,
        mdr=100.0 * miss / total,
        fdr=100.0 * false / total,
        accuracy=100.0 * exact / total,
        mae=mae,
    )


def count_spikes(counts: List[int], threshold: int = 2) -> int:
    """Count unrealistic jumps > threshold"""
    return sum(1 for i in range(1, len(counts)) if abs(counts[i] - counts[i-1]) > threshold)


def get_smoother_instance(method_name: str, fps: int = 4) -> Optional[TemporalSmoother]:
    """Get a smoother instance by name"""
    smoothers_available = {
        "Raw": None,
        "ES(α=0.2)": ExponentialSmoother(alpha=0.2),
        "ES(α=0.3)": ExponentialSmoother(alpha=0.3),
        "ES(α=0.4)": ExponentialSmoother(alpha=0.4),
        "ES(α=0.5)": ExponentialSmoother(alpha=0.5),
        "Median(w=3)": MedianFilter(window_size=3),
        "Median(w=5)": MedianFilter(window_size=5),
        "Median(w=7)": MedianFilter(window_size=7),
        "Vote(w=5,t=1)": MajorityVoting(window_size=5, tolerance=1),
        "Vote(w=7,t=1)": MajorityVoting(window_size=7, tolerance=1),
        "Vote(w=7,t=2)": MajorityVoting(window_size=7, tolerance=2),
        "Combined(w=3,α=0.3)": CombinedSmoother(median_window=3, alpha=0.3),
        "Combined(w=5,α=0.3)": CombinedSmoother(median_window=5, alpha=0.3),
        "Combined(w=5,α=0.4)": CombinedSmoother(median_window=5, alpha=0.4),
        "Delayed(mode)": DelayedAggregation(frames_per_second=fps, method="mode"),
        "Delayed(median)": DelayedAggregation(frames_per_second=fps, method="median"),
    }
    return smoothers_available.get(method_name)


def create_comparison_plots_by_gt(
    all_data: List[Dict[str, Any]],
    method_name: str,
    fps: int = 4,
    output_dir: Optional[str] = None
):
    """Create separate plots for each GT category (0, 1, 2, 3+ people)"""
    gt_groups = {}
    for data in all_data:
        gt = data['ground_truth']
        gt_label = f"{gt}_person" if gt <= 3 else "3+_person"
        if gt_label not in gt_groups:
            gt_groups[gt_label] = []
        gt_groups[gt_label].append(data)

    print("\n" + "="*100)
    print(f"VISUALIZATION: Creating plots grouped by Ground Truth")
    print("="*100)

    for gt_label, data_list in sorted(gt_groups.items()):
        n_files = len(data_list)
        gt_value = data_list[0]['ground_truth']
        print(f"\nCreating plot for GT={gt_label} ({n_files} files)")

        n_cols = min(2, n_files)
        n_rows = (n_files + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))

        if n_files == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, data in enumerate(data_list):
            ax = axes[idx]
            raw = data['raw_counts']
            smoothed = data['smoothed_counts']
            time_axis = np.arange(len(raw)) / fps

            ax.plot(time_axis, raw, 'o-', label='Raw', alpha=0.6, linewidth=1.5, markersize=3)
            ax.plot(time_axis, smoothed, 's-', label=method_name, alpha=0.8,
                   linewidth=2, markersize=3)
            ax.axhline(y=gt_value, color='g', linestyle='--', linewidth=2,
                      label=f'GT ({gt_value})')

            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('People Count', fontsize=10)
            ax.set_title(data['file_name'], fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.5, max(max(raw), max(smoothed), gt_value) + 1)

        for idx in range(n_files, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Ground Truth: {gt_label} - {method_name} vs Raw',
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 
                f'comparison_GT_{gt_label}_{method_name.replace("(", "_").replace(")", "").replace("=", "-").replace(",", "_")}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")

        plt.show()


def apply_smoothing_analysis(
    files: List[str],
    labels: List[str],
    selected_method: str,
    fps: int = 4,
    output_dir: Optional[str] = None,
    create_visualizations: bool = True
):
    """
    Apply selected smoothing method to radar data files and evaluate performance.

    Args:
        files: List of paths to radar JSON files
        labels: List of ground truth labels (e.g., "1_person", "2_person")
        selected_method: Name of smoothing method to apply
        fps: Frame rate in frames per second
        output_dir: Directory to save results
        create_visualizations: Whether to create visualization plots

    Returns:
        Tuple of (method_name, metrics_dict, all_data_for_viz)
    """
    if len(files) != len(labels):
        raise ValueError("files and labels must have the same length.")

    print("=" * 100)
    print(f"RADAR TEMPORAL SMOOTHING ANALYSIS")
    print(f"Files: {len(files)}")
    print(f"Selected Method: {selected_method}")
    print(f"Frame Rate: {fps} FPS")
    print("=" * 100)

    print("\nLoading data...")
    all_raw_counts = []
    all_ground_truths = []
    file_names = []

    for fp, gt_label in zip(files, labels):
        data = load_radar_json(fp)
        frames = data.get("data", [])
        raw_counts = extract_track_counts(frames)
        gt_count = parse_people_label(gt_label)

        all_raw_counts.append(raw_counts)
        all_ground_truths.append(gt_count)
        file_names.append(os.path.basename(fp))

    best_smoother = get_smoother_instance(selected_method, fps)
    if best_smoother is None and selected_method != "Raw":
        print(f"\nWarning: Selected method '{selected_method}' not found. Using Raw data.")
        selected_method = "Raw"
        best_smoother = None

    print(f"\nApplying method: {selected_method}")

    all_smoothed = []
    for raw_counts in all_raw_counts:
        if best_smoother is None:
            smoothed = raw_counts.copy()
        else:
            best_smoother.reset()
            smoothed = [best_smoother.smooth(c) for c in raw_counts]
        all_smoothed.extend(smoothed)

    all_true_concat = []
    for raw_counts, gt in zip(all_raw_counts, all_ground_truths):
        all_true_concat.extend([gt] * len(raw_counts))

    best_metrics = compute_metrics(all_smoothed, all_true_concat)

    total_spikes = 0
    raw_spikes = 0
    for raw_counts in all_raw_counts:
        if best_smoother is None:
            smoothed = raw_counts.copy()
        else:
            best_smoother.reset()
            smoothed = [best_smoother.smooth(c) for c in raw_counts]
        total_spikes += count_spikes(smoothed)
        raw_spikes += count_spikes(raw_counts)

    spike_reduction = 100.0 * (1 - total_spikes / raw_spikes) if raw_spikes > 0 else 0.0
    best_metrics['spikes'] = total_spikes
    best_metrics['spike_reduction'] = spike_reduction

    print("\n" + "="*100)
    print(f"📊 Performance of Selected Method: {selected_method}")
    print(f"   GFR: {best_metrics['gfr']:.2f}%")
    print(f"   GFR±1: {best_metrics['gfr1']:.2f}%")
    print(f"   F1-Weighted: {best_metrics['f1_weighted']:.2f}%")
    print(f"   MAE: {best_metrics['mae']:.2f} people")
    print(f"   Spike Reduction: {best_metrics['spike_reduction']:.2f}%")
    print("="*100)

    print("\n" + "="*100)
    print(f"APPLYING {selected_method} TO ALL FILES")
    print("="*100)

    all_data_for_viz = []
    per_file_results = []

    for i, (raw_counts, gt, fname) in enumerate(zip(all_raw_counts, all_ground_truths, file_names)):
        if best_smoother is None:
            smoothed_counts = raw_counts.copy()
        else:
            best_smoother.reset()
            smoothed_counts = [best_smoother.smooth(c) for c in raw_counts]

        true_seq = [gt] * len(raw_counts)
        metrics = compute_metrics(smoothed_counts, true_seq)

        all_data_for_viz.append({
            'file_name': fname,
            'ground_truth': gt,
            'raw_counts': raw_counts,
            'smoothed_counts': smoothed_counts,
            'metrics': metrics,
        })

        per_file_results.append({
            'File': fname,
            'Ground_Truth': gt,
            'Method': selected_method,
            'GFR (%)': metrics['gfr'],
            'GFR±1 (%)': metrics['gfr1'],
            'F1-Weighted (%)': metrics['f1_weighted'],
            'MAE': metrics['mae'],
            'Total_Frames': len(raw_counts),
        })

        print(f"{i+1}. {fname} (GT={gt}): GFR={metrics['gfr']:.1f}%, "
              f"F1={metrics['f1_weighted']:.1f}%, MAE={metrics['mae']:.2f}")

    if create_visualizations and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print("\nCreating per-GT-category comparison plots...")
        create_comparison_plots_by_gt(all_data_for_viz, selected_method, fps, output_dir)

    if output_dir:
        print("\nExporting detailed results...")
        per_file_df = pd.DataFrame(per_file_results)
        per_file_df.to_csv(
            os.path.join(output_dir, 
                f'per_file_{selected_method.replace("(", "_").replace(")", "").replace("=", "-").replace(",", "_")}_results.csv'),
            index=False
        )

        with open(os.path.join(output_dir, 'SUMMARY_REPORT.txt'), 'w') as f:
            f.write("="*100 + "\n")
            f.write("RADAR TEMPORAL SMOOTHING ANALYSIS - SUMMARY REPORT\n")
            f.write("="*100 + "\n\n")
            f.write(f"Total Files Analyzed: {len(files)}\n")
            f.write(f"Selected Method: {selected_method}\n")
            f.write(f"Frame Rate: {fps} FPS\n\n")
            f.write(f"OVERALL PERFORMANCE:\n")
            f.write(f"   GFR: {best_metrics.get('gfr', 0.0):.2f}%\n")
            f.write(f"   GFR±1: {best_metrics.get('gfr1', 0.0):.2f}%\n")
            f.write(f"   F1-Weighted: {best_metrics.get('f1_weighted', 0.0):.2f}%\n")
            f.write(f"   MAE: {best_metrics.get('mae', 0.0):.2f} people\n")
            f.write(f"   Spike Reduction: {best_metrics.get('spike_reduction', 0.0):.2f}%\n\n")
            f.write(f"PER-FILE RESULTS:\n")
            f.write(per_file_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

        print(f"\n✅ All results exported to: {output_dir}")

    print("\n" + "="*100)
    print("FINAL SUMMARY")
    print("="*100)
    print(f"\n🏆 Method: {selected_method}")
    print(f"\n📊 Overall Performance (Combined Dataset):")
    print(f"   • GFR (Exact Match): {best_metrics.get('gfr', 0.0):.2f}%")
    print(f"   • GFR±1 (Within 1): {best_metrics.get('gfr1', 0.0):.2f}%")
    print(f"   • F1-Weighted Score: {best_metrics.get('f1_weighted', 0.0):.2f}%")
    print(f"   • Mean Absolute Error: {best_metrics.get('mae', 0.0):.2f} people")
    print(f"   • Spike Reduction: {best_metrics.get('spike_reduction', 0.0):.2f}%")

    avg_file_gfr = np.mean([d['metrics']['gfr'] for d in all_data_for_viz])
    print(f"\n📈 Average Per-File Performance:")
    print(f"   • Average GFR: {avg_file_gfr:.2f}%")
    print(f"   • Files Analyzed: {len(files)}")
    print("\n" + "="*100)

    return selected_method, best_metrics, all_data_for_viz


if __name__ == "__main__":
    files = [
        "sample/data1.json",
        "sample/data2.json",
        "sample/data3.json",
    ]

    labels = [
        "1_person",
        "2_person",
        "3_person",
    ]

    selected_method_name = "Delayed(mode)"

    best_method, metrics, viz_data = apply_smoothing_analysis(
        files=files,
        labels=labels,
        selected_method=selected_method_name,
        fps=4,
        output_dir=f"./results/{selected_method_name.replace('(', '_').replace(')', '').replace('=', '-').replace(',', '_')}",
        create_visualizations=True
    )
