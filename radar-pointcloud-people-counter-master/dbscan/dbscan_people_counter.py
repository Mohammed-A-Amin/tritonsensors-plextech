from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    precision_recall_fscore_support,
    confusion_matrix,
    f1_score,
)

# =========================
# CONFIGURATION (edit here)
# =========================

# ========== OPERATION MODE ==========
USE_FIXED_PARAMS = False  # <<< CHANGE THIS: Set to False to run parameter tuning, True for run based on fixed parameters

# ========== FIXED PARAMETERS ==========
FIXED_EPS = 0.3
FIXED_MIN_SAMPLES = 5

# ========== SPATIAL FILTERING ==========
ENABLE_SPATIAL_FILTER = True  # Set to False to disable spatial filtering
X_LIM = [-1.5, 1.5]  # Min and max X coordinate limits [min, max]
Y_LIM = [0.0, 2.7]   # Min and max Y coordinate limits [min, max]
Z_LIM = [-3.0, 3.0]  # Min and max Z coordinate limits [min, max]

# ========== VELOCITY FILTERING ==========
ENABLE_VELOCITY_FILTER = True  # Set to False to disable velocity filtering
VELOCITY_THRESHOLD = 0.01  # Minimum absolute velocity to keep point (m/s)
                          # Set to 0.0 to remove only zero-velocity points

# Input files and labels
INPUT_FILES_AND_LABELS = [
    (r"/path/to/your/radar_data_3people.json", "3_person"),
    (r"path/to/your/radar_data_1person.json", "1_person"),
]

OUTPUT_DIR = "outputs"

# Frame filtering
SKIP_FIRST = 0
SKIP_LAST = 0

# DBSCAN settings
CLUSTER_MIN_SIZE_FOR_COUNT = 1
NUM_FRAMES_FOR_DBSCAN = 4

# Parameter search mode
SEARCH_MODE = "auto"  # "auto" or "grid"

# Auto mode settings
AUTO_MIN_SAMPLES_CANDIDATES = [0, 3, 4, 5, 6, 8]
AUTO_TOPK_EPS_AGG = "p50"
AUTO_SCORE_METRIC = "f1_macro"  # Options: "gfr", "gfr1", "gfr2", "f1_macro", "f1_weighted"

# Grid mode settings
GRID_EPS_LIST = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1.0]
GRID_MIN_SAMPLES_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
GRID_SCORE_METRIC = "f1_macro"  # Options: "gfr", "gfr1", "gfr2", "f1_macro", "f1_weighted"

# Radar mounting/orientation
MOUNTING = "ceiling"
VERTICAL_AXIS = "y"
RADAR_HEIGHT_M = 2.7
APPLY_FLOOR_ELEVATION = True

# DBSCAN feature selection
# USE_FEATURES = ("x", "z")
USE_FEATURES = ("x", "y", "z")

# Saving details
SAVE_COMBINED_SUMMARY_CSV = "combined_frame_summary.csv"
SAVE_SEARCH_REPORT_JSON = "search_report.json"
SAVE_PER_FILE_METRICS_CSV = "per_file_metrics.csv"
SAVE_COMBINED_METRICS_JSON = "combined_metrics.json"
SAVE_PER_GT_METRICS_CSV = "per_gt_metrics.csv"
SAVE_DETAILED_RESULTS_JSON = "detailed_results.json"

# =========================
# UTILITIES
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

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

def load_radar_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data or "data" not in data:
            return None
        return data
    except Exception:
        return None

def _apply_elevation_axis(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust the elevation axis in the point cloud to floor-referenced elevation."""
    if df.empty:
        df["elev"] = np.array([], dtype=float)
        return df
    axis = VERTICAL_AXIS.lower()
    if axis not in ("x", "y", "z"):
        axis = "y"
    if APPLY_FLOOR_ELEVATION and MOUNTING.lower() == "ceiling":
        if axis in df.columns:
            df[axis] = RADAR_HEIGHT_M - df[axis].astype(float).values
    df["elev"] = df[axis].astype(float).values if axis in df.columns else np.nan
    return df

def apply_spatial_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter points based on spatial limits."""
    if df.empty or not ENABLE_SPATIAL_FILTER:
        return df
    
    mask = np.ones(len(df), dtype=bool)
    
    # Apply X limits
    if "x" in df.columns:
        mask &= (df["x"] >= X_LIM[0]) & (df["x"] <= X_LIM[1])
    
    # Apply Y limits
    if "y" in df.columns:
        mask &= (df["y"] >= Y_LIM[0]) & (df["y"] <= Y_LIM[1])
    
    # Apply Z limits
    if "z" in df.columns:
        mask &= (df["z"] >= Z_LIM[0]) & (df["z"] <= Z_LIM[1])
    
    return df[mask].reset_index(drop=True)

def apply_velocity_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter points based on velocity threshold."""
    if df.empty or not ENABLE_VELOCITY_FILTER:
        return df
    
    if "velocity" not in df.columns:
        return df
    
    # Keep points with absolute velocity above threshold
    mask = np.abs(df["velocity"]) >= VELOCITY_THRESHOLD
    return df[mask].reset_index(drop=True)

def process_point_cloud(frame_data: Dict[str, Any]) -> pd.DataFrame:
    raw = np.array(frame_data.get("pointCloud", []), dtype=float)
    if raw.size == 0:
        return pd.DataFrame(columns=["x", "y", "z", "velocity", "snr", "noise", "trackidx",
                                     "istracked", "pointtype", "intensity", "elev"])
    
    df = pd.DataFrame(raw, columns=["x", "y", "z", "velocity", "snr", "noise", "trackidx"])
    df["istracked"] = (df["trackidx"].astype(int) != 255)
    df["pointtype"] = np.where(df["istracked"], "Tracked Point", "Point Cloud")
    df["intensity"] = df["snr"].astype(float)
    df = _apply_elevation_axis(df)
    
    # Apply filters
    df = apply_spatial_filter(df)
    df = apply_velocity_filter(df)
    
    return df[["x", "y", "z", "velocity", "snr", "noise", "trackidx",
               "istracked", "pointtype", "intensity", "elev"]]

def preprocess_all_frames(radar_data: Dict[str, Any]) -> List[pd.DataFrame]:
    frames = radar_data.get("data", [])
    points_list: List[pd.DataFrame] = []
    for frame in frames:
        fd = frame.get("frameData", {})
        points_list.append(process_point_cloud(fd))
    return points_list

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

def build_feature_matrix(df: pd.DataFrame, cols: Tuple[str, ...]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if df is None or df.empty:
        return np.empty((0, max(2, len(cols))), dtype=float), None
    cols = tuple(cols)
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Requested feature '{c}' not found in columns: {df.columns.tolist()}")
    X = df.loc[:, list(cols)].to_numpy(dtype=float)
    intens = df["intensity"].to_numpy(dtype=float) if "intensity" in df.columns else None
    return X, intens

# =========================
# METRICS
# =========================

def calculate_accuracy_metrics(predictions: List[int], truelabels: List[int]) -> Dict[str, Any]:
    total = len(predictions)
    if total == 0:
        return {
            "overallaccuracy": 0.0, "gfr": 0.0, "gfr1": 0.0, "gfr2": 0.0,
            "mdrframebased": 0.0, "fdrframebased": 0.0, "mae": 0.0,
            "f1_macro": 0.0, "f1_weighted": 0.0,
            "framestats": {"totalframes": 0, "exactmatches": 0, "within1matches": 0,
                          "within2matches": 0, "missframes": 0, "falseframes": 0},
            "predictions": predictions, "truelabels": truelabels,
        }
    
    exact = sum(1 for p, t in zip(predictions, truelabels) if p == t)
    within1 = sum(1 for p, t in zip(predictions, truelabels) if abs(p - t) <= 1)
    within2 = sum(1 for p, t in zip(predictions, truelabels) if abs(p - t) <= 2)
    miss_frames = sum(1 for p, t in zip(predictions, truelabels) if p < t)
    false_frames = sum(1 for p, t in zip(predictions, truelabels) if p > t)
    mae = np.mean([abs(p - t) for p, t in zip(predictions, truelabels)])
    
    # Calculate F1 scores
    labels = sorted(set(predictions) | set(truelabels))
    f1_macro = f1_score(truelabels, predictions, labels=labels, average='macro', zero_division=0) * 100.0
    f1_weighted = f1_score(truelabels, predictions, labels=labels, average='weighted', zero_division=0) * 100.0
    
    return {
        "overallaccuracy": exact / total * 100.0,
        "gfr": exact / total * 100.0,
        "gfr1": within1 / total * 100.0,
        "gfr2": within2 / total * 100.0,
        "mdrframebased": miss_frames / total * 100.0,
        "fdrframebased": false_frames / total * 100.0,
        "mae": mae,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "framestats": {
            "totalframes": total, "exactmatches": exact, "within1matches": within1,
            "within2matches": within2, "missframes": miss_frames, "falseframes": false_frames,
        },
        "predictions": predictions, "truelabels": truelabels,
    }

def calculate_per_class_metrics(predictions: List[int], truelabels: List[int]) -> Dict[str, Any]:
    labels = sorted(set(predictions) | set(truelabels))
    if not labels:
        return {"perclassmetrics": pd.DataFrame(), "confusionmatrix": np.array([]), "classlabels": []}
    
    precision, recall, f1, support = precision_recall_fscore_support(
        truelabels, predictions, labels=labels, average=None, zero_division=0
    )
    
    rows = []
    for i, c in enumerate(labels):
        rows.append({
            "People Count": c,
            "Precision": precision[i] * 100.0,
            "Recall": recall[i] * 100.0,
            "F1-Score": f1[i] * 100.0,
            "Support Frames": int(support[i]),
        })
    
    cm = confusion_matrix(truelabels, predictions, labels=labels)
    return {"perclassmetrics": pd.DataFrame(rows), "confusionmatrix": cm, "classlabels": labels}

def calculate_metrics_from_counts(pred_counts: List[int], true_counts: List[int]) -> Dict[str, Any]:
    """Calculate comprehensive metrics given predictions and true labels."""
    if len(pred_counts) == 0:
        return {
            "overallaccuracy": 0.0, "gfr": 0.0, "gfr1": 0.0, "gfr2": 0.0,
            "mdrframebased": 0.0, "fdrframebased": 0.0, "mae": 0.0,
            "f1_macro": 0.0, "f1_weighted": 0.0,
            "perclassmetrics": pd.DataFrame(), "confusionmatrix": np.array([]),
            "classlabels": [], "predictions": [], "truelabels": [],
            "framestats": {},
        }
    
    met = calculate_accuracy_metrics(pred_counts, true_counts)
    pc = calculate_per_class_metrics(pred_counts, true_counts)
    met.update({
        "perclassmetrics": pc["perclassmetrics"],
        "confusionmatrix": pc["confusionmatrix"],
        "classlabels": pc["classlabels"]
    })
    return met

# =========================
# DBSCAN
# =========================

def estimate_eps_kdistance(X: np.ndarray, k: int = 4) -> float:
    if X is None or len(X) == 0:
        return 0.0
    k = max(1, int(k))
    try:
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(X)
        distances, _ = neigh.kneighbors(X)
        kdist = np.sort(distances[:, k - 1])
        y = kdist
        if len(y) < 6:
            return float(np.percentile(kdist, 90))
        
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
        dy = np.gradient(y_norm)
        ddy = np.gradient(dy)
        knee_idx = int(np.argmax(ddy))
        eps_cand = float(kdist[knee_idx])
        
        if eps_cand <= 0 or not np.isfinite(eps_cand):
            return float(np.percentile(kdist, 90))
        
        if eps_cand < 0.001:
            d = np.linalg.norm(X - X.mean(axis=0), axis=1)
            scale = float(np.percentile(d, 75)) if len(d) > 0 else 0.05
            return max(0.01, scale * 0.05)
        
        return eps_cand
    except Exception:
        d = np.linalg.norm(X - X.mean(axis=0), axis=1) if len(X) else np.array([0.0])
        return float(np.percentile(d, 75)) if len(d) > 0 else 0.05

def run_dbscan_combined_frames(
    X_all: List[np.ndarray],
    intens_all: List[Optional[np.ndarray]],
    from_idx: int,
    to_idx: int,
    minsamples_user: int,
    eps_override: float,
    feature_names: Tuple[str, ...] = ("x", "z")
) -> Dict[str, Any]:
    """Run DBSCAN on combined frames from from_idx to to_idx (inclusive)."""
    combined_X_list = [X_all[i] for i in range(from_idx, to_idx + 1)
                       if 0 <= i < len(X_all) and X_all[i] is not None and X_all[i].size > 0]
    
    if not combined_X_list:
        return {"success": False, "reason": "empty_combined", "npoints": 0,
                "clustersizes": {}, "labels": np.array([], dtype=int)}
    
    X_combined = np.vstack(combined_X_list)
    combined_intens_list = [intens_all[i] for i in range(from_idx, to_idx + 1)
                            if 0 <= i < len(intens_all) and intens_all[i] is not None
                            and intens_all[i].size > 0]
    intens_combined = np.concatenate(combined_intens_list) if combined_intens_list else None
    
    if minsamples_user == 0:
        minsamples = max(3, 2 * X_combined.shape[1])
    else:
        minsamples = int(minsamples_user)
    
    eps = float(eps_override)
    db = DBSCAN(eps=eps, min_samples=minsamples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(X_combined)
    
    unique = set(labels)
    nclusters = len([l for l in unique if l != -1])
    nnoise = int((labels == -1).sum())
    
    clustersizes: Dict[int, int] = {}
    centroids: Dict[int, Dict[str, float]] = {}
    avgsnr: Dict[int, Optional[float]] = {}
    
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        mask = (labels == lab)
        clustersizes[int(lab)] = int(mask.sum())
        cent = X_combined[mask].mean(axis=0)
        centroids[int(lab)] = {name: float(val) for name, val in zip(feature_names, cent)}
        avgsnr[int(lab)] = float(np.nanmean(intens_combined[mask])) if intens_combined is not None and mask.any() else None
    
    return {
        "success": True,
        "npoints": int(X_combined.shape[0]),
        "nclusters": int(nclusters),
        "nnoise": int(nnoise),
        "clustersizes": clustersizes,
        "centroids": centroids,
        "avgsnr": avgsnr,
        "eps": float(eps),
        "minsamples": int(minsamples),
        "labels": labels.astype(int),
    }

def count_people_from_clusters(cluster_sizes: Dict[int, int], min_size: int) -> int:
    return int(sum(1 for s in cluster_sizes.values() if s >= int(min_size)))

# =========================
# PARAMETER SEARCH
# =========================

def evaluate_counts_to_metric(counts: List[int], target: List[int], metric: str) -> float:
    """
    Evaluate predictions against ground truth using specified metric.
    
    Args:
        counts: Predicted people counts
        target: Ground truth people counts
        metric: One of "gfr", "gfr1", "gfr2", "f1_macro", "f1_weighted"
    
    Returns:
        Score value (higher is better)
    """
    met = calculate_metrics_from_counts(counts, target)
    metric = metric.lower()
    
    if metric == "gfr":
        return float(met["gfr"])
    elif metric == "gfr1":
        return float(met["gfr1"])
    elif metric == "gfr2":
        return float(met["gfr2"])
    elif metric == "f1_macro":
        return float(met["f1_macro"])
    elif metric == "f1_weighted":
        return float(met["f1_weighted"])
    else:
        # Default to GFR if unknown metric
        return float(met["gfr"])

def aggregate_eps(values: List[float], mode: str) -> float:
    arr = np.array([v for v in values if np.isfinite(v) and v > 0.0], dtype=float)
    if arr.size == 0:
        return 0.2
    mode = mode.lower()
    if mode in ("median", "p50"):
        return float(np.percentile(arr, 50))
    if mode == "p75":
        return float(np.percentile(arr, 75))
    if mode == "mean":
        return float(np.mean(arr))
    return float(np.percentile(arr, 50))

def search_params_grid(
    X_all: List[np.ndarray],
    intens_all: List[Optional[np.ndarray]],
    y_target: List[int],
    eps_list: List[float],
    ms_list: List[int],
    score_metric: str
) -> Tuple[float, int, Dict[str, Any]]:
    """Grid search over eps and min_samples."""
    best_score = -np.inf
    best = {"eps": None, "min_samples": None, "score": None, "gfr": None, "gfr1": None, 
            "gfr2": None, "f1_macro": None, "f1_weighted": None, "mdr": None, "fdr": None}
    records = []
    y_target_eval = y_target[NUM_FRAMES_FOR_DBSCAN - 1:]
    
    print(f"  Optimizing for metric: {score_metric.upper()}")
    print(f"  Testing {len(eps_list)} eps values × {len(ms_list)} min_samples values = {len(eps_list) * len(ms_list)} combinations")
    
    for ms in ms_list:
        for eps in eps_list:
            pred_counts = []
            for i in range(NUM_FRAMES_FOR_DBSCAN - 1, len(X_all)):
                res = run_dbscan_combined_frames(
                    X_all, intens_all, i - NUM_FRAMES_FOR_DBSCAN + 1, i,
                    minsamples_user=int(ms), eps_override=float(eps), feature_names=USE_FEATURES
                )
                c = count_people_from_clusters(res.get("clustersizes", {}), CLUSTER_MIN_SIZE_FOR_COUNT) if res.get("success", False) else 0
                pred_counts.append(int(c))
            
            score = evaluate_counts_to_metric(pred_counts, y_target_eval, score_metric)
            m_all = calculate_metrics_from_counts(pred_counts, y_target_eval)
            rec = {
                "min_samples": int(ms),
                "eps": float(eps),
                "score": float(score),
                "metric_optimized": score_metric,
                "gfr": float(m_all["gfr"]),
                "gfr1": float(m_all["gfr1"]),
                "gfr2": float(m_all["gfr2"]),
                "f1_macro": float(m_all["f1_macro"]),
                "f1_weighted": float(m_all["f1_weighted"]),
                "mdr": float(m_all["mdrframebased"]),
                "fdr": float(m_all["fdrframebased"]),
            }
            records.append(rec)
            
            if score > best_score:
                best_score = score
                best.update(rec)
    
    print(f"  ✓ Best score: {best_score:.2f}% (eps={best['eps']:.4f}, min_samples={best['min_samples']})")
    
    return float(best["eps"]), int(best["min_samples"]), {"search": records, "best": best}

def search_params_auto(
    X_all: List[np.ndarray],
    intens_all: List[Optional[np.ndarray]],
    y_target: List[int],
    min_samples_candidates: List[int],
    eps_agg: str,
    score_metric: str
) -> Tuple[float, int, Dict[str, Any]]:
    """Auto search: estimate eps per min_samples candidate."""
    best_score = -np.inf
    best = {"eps": None, "min_samples": None, "score": None, "gfr": None, "gfr1": None,
            "gfr2": None, "f1_macro": None, "f1_weighted": None, "mdr": None, "fdr": None}
    per_ms_records = []
    y_target_eval = y_target[NUM_FRAMES_FOR_DBSCAN - 1:]
    
    print(f"  Optimizing for metric: {score_metric.upper()}")
    print(f"  Testing {len(min_samples_candidates)} min_samples candidates")
    
    ms_list = []
    for ms in min_samples_candidates:
        if ms == 0:
            d = next((X.shape[1] for X in X_all if X is not None and len(X) and X.ndim == 2), 2)
            ms_list.append(max(3, 2 * d))
        else:
            ms_list.append(int(ms))
    
    for ms in ms_list:
        eps_list = [estimate_eps_kdistance(X, k=ms) if X is not None and len(X) > 0 else np.nan
                    for X in X_all]
        eps_global = aggregate_eps(eps_list, eps_agg)
        
        pred_counts = []
        for i in range(NUM_FRAMES_FOR_DBSCAN - 1, len(X_all)):
            res = run_dbscan_combined_frames(
                X_all, intens_all, i - NUM_FRAMES_FOR_DBSCAN + 1, i,
                minsamples_user=ms, eps_override=eps_global, feature_names=USE_FEATURES
            )
            c = count_people_from_clusters(res.get("clustersizes", {}), CLUSTER_MIN_SIZE_FOR_COUNT) if res.get("success", False) else 0
            pred_counts.append(int(c))
        
        score = evaluate_counts_to_metric(pred_counts, y_target_eval, score_metric)
        m_all = calculate_metrics_from_counts(pred_counts, y_target_eval)
        rec = {
            "min_samples": int(ms),
            "eps": float(eps_global),
            "score": float(score),
            "metric_optimized": score_metric,
            "gfr": float(m_all["gfr"]),
            "gfr1": float(m_all["gfr1"]),
            "gfr2": float(m_all["gfr2"]),
            "f1_macro": float(m_all["f1_macro"]),
            "f1_weighted": float(m_all["f1_weighted"]),
            "mdr": float(m_all["mdrframebased"]),
            "fdr": float(m_all["fdrframebased"]),
        }
        per_ms_records.append(rec)
        
        if score > best_score:
            best_score = score
            best.update(rec)
    
    print(f"  ✓ Best score: {best_score:.2f}% (eps={best['eps']:.4f}, min_samples={best['min_samples']})")
    
    return float(best["eps"]), int(best["min_samples"]), {"search": per_ms_records, "best": best}

# =========================
# MAIN PIPELINE
# =========================

def main():
    ensure_dir(OUTPUT_DIR)
    
    if not INPUT_FILES_AND_LABELS:
        raise ValueError("INPUT_FILES_AND_LABELS is empty. Please add files and labels.")
    
    print(f"\n{'='*80}")
    print(f"DBSCAN PEOPLE COUNTING PIPELINE")
    print(f"{'='*80}")
    print(f"Mode: {'INFERENCE (Fixed Parameters)' if USE_FIXED_PARAMS else 'PARAMETER SEARCH'}")
    print(f"Processing {len(INPUT_FILES_AND_LABELS)} file(s)...")
    print(f"\n--- Filtering Configuration ---")
    print(f"Spatial Filter: {'ENABLED' if ENABLE_SPATIAL_FILTER else 'DISABLED'}")
    if ENABLE_SPATIAL_FILTER:
        print(f"  X limits: {X_LIM}")
        print(f"  Y limits: {Y_LIM}")
        print(f"  Z limits: {Z_LIM}")
    print(f"Velocity Filter: {'ENABLED' if ENABLE_VELOCITY_FILTER else 'DISABLED'}")
    if ENABLE_VELOCITY_FILTER:
        print(f"  Velocity threshold: {VELOCITY_THRESHOLD} m/s")
    
    if not USE_FIXED_PARAMS:
        metric = AUTO_SCORE_METRIC if SEARCH_MODE.lower() == "auto" else GRID_SCORE_METRIC
        print(f"\n--- Optimization Configuration ---")
        print(f"Search mode: {SEARCH_MODE.upper()}")
        print(f"Optimization metric: {metric.upper()}")
    
    print(f"{'='*80}\n")
    
    # === STEP 1: Load and aggregate all frames ===
    all_X_kept = []
    all_intens_kept = []
    all_y_target = []
    all_file_indices = []
    file_info = []
    
    total_points_before = 0
    total_points_after = 0
    
    for file_idx, (file_path, gt_label) in enumerate(INPUT_FILES_AND_LABELS):
        print(f"Loading: {os.path.basename(file_path)} (GT: {gt_label})")
        radar_data = load_radar_json_file(file_path)
        
        if radar_data is None:
            print(f"  ⚠ Skipping invalid file: {file_path}")
            continue
        
        points_list_all = preprocess_all_frames(radar_data)
        total_frames = len(points_list_all)
        
        if total_frames == 0:
            print(f"  ⚠ Skipping file with no frames: {file_path}")
            continue
        
        kept_indices = filter_edge_frames(total_frames, SKIP_FIRST, SKIP_LAST)
        
        if len(kept_indices) < NUM_FRAMES_FOR_DBSCAN:
            print(f"  ⚠ Skipping file with insufficient frames after filtering: {file_path}")
            continue
        
        try:
            gt_count = parse_people_label(gt_label)
        except ValueError as e:
            print(f"  ⚠ Skipping file with invalid label: {e}")
            continue
        
        file_points_before = 0
        file_points_after = 0
        
        for idx in kept_indices:
            # Count points before building feature matrix (already filtered in process_point_cloud)
            file_points_after += len(points_list_all[idx])
            
            X, intens = build_feature_matrix(points_list_all[idx], USE_FEATURES)
            all_X_kept.append(X)
            all_intens_kept.append(intens)
            all_y_target.append(gt_count)
            all_file_indices.append(file_idx)
        
        total_points_after += file_points_after
        
        file_info.append({
            "file": os.path.basename(file_path),
            "path": file_path,
            "gt_label": gt_label,
            "gt_count": gt_count,
            "total_frames": total_frames,
            "kept_frames": len(kept_indices),
            "start_idx": len(all_y_target) - len(kept_indices),
            "end_idx": len(all_y_target),
            "file_index": file_idx,
        })
        
        print(f"  ✓ Loaded {len(kept_indices)} frames (GT={gt_count})")
        print(f"    Points after filtering: {file_points_after}")
    
    if len(all_X_kept) < NUM_FRAMES_FOR_DBSCAN:
        raise RuntimeError(f"Insufficient total frames ({len(all_X_kept)}) across all files.")
    
    print(f"\n{'='*80}")
    print(f"COMBINED DATASET: {len(all_X_kept)} frames from {len(file_info)} files")
    if ENABLE_SPATIAL_FILTER or ENABLE_VELOCITY_FILTER:
        print(f"Total points after filtering: {total_points_after}")
    print(f"{'='*80}\n")
    
    # === STEP 2: Determine parameters ===
    if USE_FIXED_PARAMS:
        print(f"Using FIXED parameters:")
        print(f"  eps = {FIXED_EPS}")
        print(f"  min_samples = {FIXED_MIN_SAMPLES}")
        best_eps = FIXED_EPS
        best_ms = FIXED_MIN_SAMPLES
        search_report = {
            "best": {
                "eps": float(best_eps),
                "min_samples": int(best_ms),
                "metric_optimized": None,
                "gfr": None,
                "gfr1": None,
                "gfr2": None,
                "f1_macro": None,
                "f1_weighted": None,
                "mdr": None,
                "fdr": None
            },
            "search": []
        }
    else:
        print(f"Running {SEARCH_MODE.upper()} parameter search on combined dataset...")
        search_func = search_params_auto if SEARCH_MODE.lower() == "auto" else search_params_grid
        search_args = {
            "X_all": all_X_kept,
            "intens_all": all_intens_kept,
            "y_target": all_y_target,
            "score_metric": AUTO_SCORE_METRIC if SEARCH_MODE.lower() == "auto" else GRID_SCORE_METRIC,
        }
        
        if SEARCH_MODE.lower() == "auto":
            search_args.update({
                "min_samples_candidates": AUTO_MIN_SAMPLES_CANDIDATES,
                "eps_agg": AUTO_TOPK_EPS_AGG
            })
        else:
            search_args.update({
                "eps_list": GRID_EPS_LIST,
                "ms_list": GRID_MIN_SAMPLES_LIST
            })
        
        best_eps, best_ms, search_report = search_func(**search_args)
        
        print(f"\n✓ Best parameters found:")
        print(f"  eps = {best_eps:.4f}")
        print(f"  min_samples = {best_ms}")
        if search_report['best'].get('gfr') is not None:
            print(f"  GFR = {search_report['best']['gfr']:.2f}%")
            print(f"  GFR±1 = {search_report['best']['gfr1']:.2f}%")
            print(f"  F1-macro = {search_report['best']['f1_macro']:.2f}%")
            print(f"  F1-weighted = {search_report['best']['f1_weighted']:.2f}%")
    
    # === STEP 3: Apply best params and collect predictions ===
    print(f"\n{'='*80}")
    print(f"APPLYING PARAMETERS TO ALL FRAMES")
    print(f"{'='*80}\n")
    
    all_pred_counts = []
    summaries = []
    detailed_frame_results = []
    
    for i in range(NUM_FRAMES_FOR_DBSCAN - 1, len(all_X_kept)):
        res = run_dbscan_combined_frames(
            all_X_kept, all_intens_kept,
            from_idx=i - NUM_FRAMES_FOR_DBSCAN + 1,
            to_idx=i,
            minsamples_user=best_ms,
            eps_override=best_eps,
            feature_names=USE_FEATURES
        )
        
        if res.get("success", False):
            npoints, nclusters, nnoise = res["npoints"], res["nclusters"], res["nnoise"]
            people_count = count_people_from_clusters(res.get("clustersizes", {}), CLUSTER_MIN_SIZE_FOR_COUNT)
            cluster_details = res.get("clustersizes", {})
            centroids = res.get("centroids", {})
        else:
            npoints, nclusters, nnoise, people_count = 0, 0, 0, 0
            cluster_details = {}
            centroids = {}
        
        all_pred_counts.append(int(people_count))
        
        # Find filename for this frame
        current_file_info = next((f for f in file_info if f["start_idx"] <= i < f["end_idx"]), None)
        filename = current_file_info["file"] if current_file_info else "unknown"
        file_gt_count = current_file_info["gt_count"] if current_file_info else all_y_target[i]
        
        summaries.append({
            "global_frame_idx": i,
            "filename": filename,
            "file_index": all_file_indices[i],
            "npoints": npoints,
            "nclusters": nclusters,
            "nnoise": nnoise,
            "dbscan_people_count": int(people_count),
            "ground_truth": file_gt_count,
            "eps": float(best_eps),
            "min_samples": int(best_ms),
        })
        
        detailed_frame_results.append({
            "frame_idx": i,
            "filename": filename,
            "predicted_count": int(people_count),
            "ground_truth": file_gt_count,
            "correct": int(people_count) == file_gt_count,
            "error": int(people_count) - file_gt_count,
            "num_points": npoints,
            "num_clusters": nclusters,
            "num_noise_points": nnoise,
            "cluster_sizes": cluster_details,
            "cluster_centroids": centroids,
            "eps_used": float(best_eps),
            "min_samples_used": int(best_ms),
        })
    
    # === STEP 4: Calculate metrics ===
    y_target_eval = all_y_target[NUM_FRAMES_FOR_DBSCAN - 1:]
    combined_metrics = calculate_metrics_from_counts(all_pred_counts, y_target_eval)
    
    print(f"\n{'='*80}")
    print(f"COMBINED METRICS (Raw DBSCAN - all files together)")
    print(f"{'='*80}")
    print(f"Total evaluated frames: {len(y_target_eval)}")
    print(f"Parameters used: eps={best_eps}, min_samples={best_ms}")
    print(f"Features used: {USE_FEATURES}")
    print(f"Frames per DBSCAN window: {NUM_FRAMES_FOR_DBSCAN}")
    print(f"\n--- Accuracy Metrics ---")
    print(f"  GFR (exact): {combined_metrics['gfr']:.2f}%")
    print(f"  GFR±1: {combined_metrics['gfr1']:.2f}%")
    print(f"  GFR±2: {combined_metrics['gfr2']:.2f}%")
    print(f"  F1-macro: {combined_metrics['f1_macro']:.2f}%")
    print(f"  F1-weighted: {combined_metrics['f1_weighted']:.2f}%")
    print(f"  MDR (Miss Detection Rate): {combined_metrics['mdrframebased']:.2f}%")
    print(f"  FDR (False Detection Rate): {combined_metrics['fdrframebased']:.2f}%")
    print(f"  MAE: {combined_metrics['mae']:.2f}")
    
    # === STEP 5: Per-file metrics ===
    print(f"\n{'='*80}")
    print(f"PER-FILE METRICS")
    print(f"{'='*80}")
    
    per_file_results = []
    for finfo in file_info:
        start = finfo["start_idx"] + NUM_FRAMES_FOR_DBSCAN - 1
        end = finfo["end_idx"]
        
        if start >= len(all_pred_counts):
            continue
        
        file_preds = all_pred_counts[start - (NUM_FRAMES_FOR_DBSCAN - 1):end - (NUM_FRAMES_FOR_DBSCAN - 1)]
        file_trues = [finfo["gt_count"]] * len(file_preds)
        
        if len(file_preds) == 0:
            continue
        
        file_metrics = calculate_metrics_from_counts(file_preds, file_trues)
        
        per_file_results.append({
            "file": finfo["file"],
            "gt_label": finfo["gt_label"],
            "gt_count": finfo["gt_count"],
            "total_frames_in_file": finfo["total_frames"],
            "kept_frames": finfo["kept_frames"],
            "evaluated_frames": len(file_preds),
            "gfr": file_metrics["gfr"],
            "gfr1": file_metrics["gfr1"],
            "gfr2": file_metrics["gfr2"],
            "f1_macro": file_metrics["f1_macro"],
            "f1_weighted": file_metrics["f1_weighted"],
            "mdr": file_metrics["mdrframebased"],
            "fdr": file_metrics["fdrframebased"],
            "mae": file_metrics["mae"],
            "exact_matches": file_metrics["framestats"]["exactmatches"],
        })
        
        print(f"\n{finfo['file']}")
        print(f"  GT: {finfo['gt_count']} people")
        print(f"  Frames: {len(file_preds)} evaluated (from {finfo['total_frames']} total)")
        print(f"  GFR: {file_metrics['gfr']:.2f}%")
        print(f"  GFR±1: {file_metrics['gfr1']:.2f}%")
        print(f"  F1-macro: {file_metrics['f1_macro']:.2f}%")
        print(f"  F1-weighted: {file_metrics['f1_weighted']:.2f}%")
        print(f"  MDR: {file_metrics['mdrframebased']:.2f}%")
        print(f"  FDR: {file_metrics['fdrframebased']:.2f}%")
        print(f"  MAE: {file_metrics['mae']:.2f}")
        print(f"  Exact matches: {file_metrics['framestats']['exactmatches']}/{len(file_preds)}")
    
    # === STEP 6: Save outputs ===
    print(f"\n{'='*80}")
    print(f"SAVING OUTPUTS")
    print(f"{'='*80}")
    
    # Frame summary CSV with filename
    df_summary = pd.DataFrame(summaries)
    csv_path = os.path.join(OUTPUT_DIR, SAVE_COMBINED_SUMMARY_CSV)
    df_summary.to_csv(csv_path, index=False)
    print(f"  ✓ {csv_path}")
    print(f"    Columns: {list(df_summary.columns)}")
    
    # Search report
    report = {
        "mode": "fixed_params" if USE_FIXED_PARAMS else "parameter_search",
        "search_mode": None if USE_FIXED_PARAMS else SEARCH_MODE,
        "optimization_metric": None if USE_FIXED_PARAMS else (AUTO_SCORE_METRIC if SEARCH_MODE.lower() == "auto" else GRID_SCORE_METRIC),
        "parameters_used": {"eps": float(best_eps), "min_samples": int(best_ms)},
        "filtering_configuration": {
            "spatial_filter_enabled": ENABLE_SPATIAL_FILTER,
            "x_limits": X_LIM if ENABLE_SPATIAL_FILTER else None,
            "y_limits": Y_LIM if ENABLE_SPATIAL_FILTER else None,
            "z_limits": Z_LIM if ENABLE_SPATIAL_FILTER else None,
            "velocity_filter_enabled": ENABLE_VELOCITY_FILTER,
            "velocity_threshold": VELOCITY_THRESHOLD if ENABLE_VELOCITY_FILTER else None,
        },
        "configuration": {
            "num_files": len(file_info),
            "total_frames_loaded": len(all_X_kept),
            "total_frames_evaluated": len(y_target_eval),
            "features_used": list(USE_FEATURES),
            "num_frames_for_dbscan": NUM_FRAMES_FOR_DBSCAN,
            "cluster_min_size_for_count": CLUSTER_MIN_SIZE_FOR_COUNT,
            "skip_first": SKIP_FIRST,
            "skip_last": SKIP_LAST,
        },
        "search_records": search_report.get("search", []) if not USE_FIXED_PARAMS else [],
        "best_from_search": search_report.get("best", {}) if not USE_FIXED_PARAMS else None,
        "combined_metrics": {k: v for k, v in combined_metrics.items()
                            if not isinstance(v, (pd.DataFrame, np.ndarray))},
    }
    
    report_path = os.path.join(OUTPUT_DIR, SAVE_SEARCH_REPORT_JSON)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {report_path}")
    
    # Per-file metrics CSV
    if per_file_results:
        df_per_file = pd.DataFrame(per_file_results)
        per_file_path = os.path.join(OUTPUT_DIR, SAVE_PER_FILE_METRICS_CSV)
        df_per_file.to_csv(per_file_path, index=False)
        print(f"  ✓ {per_file_path}")
    
    # Per-GT metrics
    print(f"\n{'='*80}")
    print(f"PER-GT GROUPED METRICS")
    print(f"{'='*80}")
    
    gt_groups = sorted(set(y_target_eval))
    per_gt_results = []
    
    for gt in gt_groups:
        mask = [t == gt for t in y_target_eval]
        preds_gt = [p for p, m in zip(all_pred_counts, mask) if m]
        trues_gt = [t for t, m in zip(y_target_eval, mask) if m]
        
        if len(trues_gt) == 0:
            continue
        
        m_gt = calculate_metrics_from_counts(preds_gt, trues_gt)
        
        per_gt_results.append({
            "GT": gt,
            "n_frames": len(trues_gt),
            "gfr": m_gt["gfr"],
            "gfr1": m_gt["gfr1"],
            "gfr2": m_gt["gfr2"],
            "f1_macro": m_gt["f1_macro"],
            "f1_weighted": m_gt["f1_weighted"],
            "mdr": m_gt["mdrframebased"],
            "fdr": m_gt["fdrframebased"],
            "mae": m_gt["mae"],
            "exact_matches": m_gt["framestats"]["exactmatches"],
        })
        
        print(f"\nGT = {gt} people")
        print(f"  Frames: {len(trues_gt)}")
        print(f"  GFR: {m_gt['gfr']:.2f}%")
        print(f"  GFR±1: {m_gt['gfr1']:.2f}%")
        print(f"  F1-macro: {m_gt['f1_macro']:.2f}%")
        print(f"  F1-weighted: {m_gt['f1_weighted']:.2f}%")
        print(f"  MDR: {m_gt['mdrframebased']:.2f}%")
        print(f"  FDR: {m_gt['fdrframebased']:.2f}%")
        print(f"  MAE: {m_gt['mae']:.2f}")
        print(f"  Exact matches: {m_gt['framestats']['exactmatches']}/{len(trues_gt)}")
    
    # Save per-GT metrics
    if per_gt_results:
        df_per_gt = pd.DataFrame(per_gt_results)
        per_gt_path = os.path.join(OUTPUT_DIR, SAVE_PER_GT_METRICS_CSV)
        df_per_gt.to_csv(per_gt_path, index=False)
        print(f"  ✓ {per_gt_path}")
    
    # Combined metrics JSON
    combined_metrics_json = {
        "combined": {k: v for k, v in combined_metrics.items()
                    if not isinstance(v, (pd.DataFrame, np.ndarray))},
        "per_file": per_file_results,
        "per_gt": per_gt_results,
        "confusion_matrix": combined_metrics["confusionmatrix"].tolist() if isinstance(combined_metrics.get("confusionmatrix"), np.ndarray) else [],
        "class_labels": combined_metrics.get("classlabels", []),
    }
    
    combined_metrics_path = os.path.join(OUTPUT_DIR, SAVE_COMBINED_METRICS_JSON)
    with open(combined_metrics_path, "w", encoding="utf-8") as f:
        json.dump(combined_metrics_json, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {combined_metrics_path}")
    
    # Detailed results with cluster information per frame
    detailed_results = {
        "configuration": report["configuration"],
        "filtering_configuration": report["filtering_configuration"],
        "optimization_metric": report.get("optimization_metric"),
        "parameters": {"eps": float(best_eps), "min_samples": int(best_ms)},
        "file_info": file_info,
        "frame_results": detailed_frame_results,
        "summary_statistics": {
            "total_frames_evaluated": len(y_target_eval),
            "total_correct_predictions": sum(1 for r in detailed_frame_results if r["correct"]),
            "total_clusters_detected": sum(r["num_clusters"] for r in detailed_frame_results),
            "total_points_processed": sum(r["num_points"] for r in detailed_frame_results),
            "average_clusters_per_frame": float(np.mean([r["num_clusters"] for r in detailed_frame_results])),
            "average_points_per_frame": float(np.mean([r["num_points"] for r in detailed_frame_results])),
        },
    }
    
    detailed_path = os.path.join(OUTPUT_DIR, SAVE_DETAILED_RESULTS_JSON)
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {detailed_path}")
    
    # Per-class metrics table
    if not combined_metrics["perclassmetrics"].empty:
        per_class_path = os.path.join(OUTPUT_DIR, "per_class_metrics.csv")
        combined_metrics["perclassmetrics"].to_csv(per_class_path, index=False)
        print(f"  ✓ {per_class_path}")
    
    print(f"\n{'='*80}")
    print(f"✓ COMPLETE - Ready for smoothing analysis!")
    print(f"{'='*80}")
    print(f"Mode: {'FIXED PARAMETERS' if USE_FIXED_PARAMS else 'PARAMETER SEARCH'}")
    print(f"Parameters: eps={best_eps:.4f}, min_samples={best_ms}")
    print(f"Filtering: Spatial={'ON' if ENABLE_SPATIAL_FILTER else 'OFF'}, Velocity={'ON' if ENABLE_VELOCITY_FILTER else 'OFF'}")
    if not USE_FIXED_PARAMS:
        metric = AUTO_SCORE_METRIC if SEARCH_MODE.lower() == "auto" else GRID_SCORE_METRIC
        print(f"Optimization metric: {metric.upper()}")
    print(f"Frames evaluated: {len(y_target_eval)}")
    print(f"Combined GFR: {combined_metrics['gfr']:.2f}%")
    print(f"Combined GFR±1: {combined_metrics['gfr1']:.2f}%")
    print(f"Combined F1-macro: {combined_metrics['f1_macro']:.2f}%")
    print(f"Combined F1-weighted: {combined_metrics['f1_weighted']:.2f}%")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"\nNext step: Run smoothing analysis on '{SAVE_COMBINED_SUMMARY_CSV}'")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
