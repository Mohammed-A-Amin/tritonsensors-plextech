# Radar Temporal Smoothing Analysis

A Python tool for applying temporal smoothing methods to radar point cloud data for people counting applications. This tool processes JSON radar data files and evaluates smoothing performance using various metrics.

## Features

- **Multiple Smoothing Methods**: Supports different temporal smoothing algorithms including exponential smoothing, median filters, majority voting, combined methods, and delayed aggregation
- **Comprehensive Metrics**: Evaluates performance using GFR (Good Frame Rate), F1 scores, MAE (Mean Absolute Error), and spike reduction metrics
- **Visualization**: Generates comparison plots grouped by ground truth categories
- **Detailed Reporting**: Exports per-file results, summary statistics, and visualization plots

## Requirements

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage

### Direct Configuration (Edit the Script)

Edit the `if __name__ == "__main__"` section at the bottom of the script:

```python
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
```

Then run:
```bash
python radar_temporal_smoothing.py
```

### Python API

```python
from radar_temporal_smoothing import apply_smoothing_analysis

method, metrics, viz_data = apply_smoothing_analysis(
    files=['sample/recording1.json', 'sample/recording2.json'],
    labels=['1_person', '2_person'],
    selected_method='Delayed(mode)',
    fps=4,
    output_dir='./results',
    create_visualizations=True
)
```

## Available Smoothing Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `Raw` | No smoothing (baseline) | - |
| `ES(α=0.2)` | Exponential smoothing | α: 0.2 |
| `ES(α=0.3)` | Exponential smoothing | α: 0.3 |
| `ES(α=0.4)` | Exponential smoothing | α: 0.4 |
| `ES(α=0.5)` | Exponential smoothing | α: 0.5 |
| `Median(w=3)` | Median filter | window: 3 |
| `Median(w=5)` | Median filter | window: 5 |
| `Median(w=7)` | Median filter | window: 7 |
| `Vote(w=5,t=1)` | Majority voting | window: 5, tolerance: 1 |
| `Vote(w=7,t=1)` | Majority voting | window: 7, tolerance: 1 |
| `Vote(w=7,t=2)` | Majority voting | window: 7, tolerance: 2 |
| `Combined(w=3,α=0.3)` | Median + Exponential | window: 3, α: 0.3 |
| `Combined(w=5,α=0.3)` | Median + Exponential | window: 5, α: 0.3 |
| `Combined(w=5,α=0.4)` | Median + Exponential | window: 5, α: 0.4 |
| `Delayed(mode)` | Delayed aggregation | mode |
| `Delayed(median)` | Delayed aggregation | median |

## Input Data Format

The tool expects JSON files with the following structure:

```json
{
  "data": [
    {
      "frameData": {
        "numDetectedTracks": 2,
        "trackData": [...]
      }
    }
  ]
}
```

Ground truth labels should follow the format: `N_person` where N is the number of people (e.g., `0_person`, `1_person`, `2_person`, `3_person`).

## Output

The tool generates:

1. **CSV Files**:
   - `per_file_{method}_results.csv`: Detailed per-file metrics

2. **Visualizations** (PNG):
   - `comparison_GT_{category}_{method}.png`: Plots grouped by ground truth

3. **Summary Report** (TXT):
   - `SUMMARY_REPORT.txt`: Overall performance metrics and per-file results

## Performance Metrics

- **GFR (Good Frame Rate)**: Percentage of frames with exact count match
- **GFR±1**: Percentage of frames within ±1 person of ground truth
- **F1-Weighted**: Weighted F1 score across all classes
- **MAE**: Mean Absolute Error in person count
- **Spike Reduction**: Reduction in unrealistic count jumps

## Function Parameters

The main function `apply_smoothing_analysis()` accepts:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `files` | List[str] | Required | Paths to radar JSON files |
| `labels` | List[str] | Required | Ground truth labels (e.g., "1_person", "2_person") |
| `selected_method` | str | Required | Smoothing method to apply |
| `fps` | int | 4 | Frame rate in frames per second |
| `output_dir` | str | None | Output directory for results |
| `create_visualizations` | bool | True | Whether to create visualization plots |

## Example Configurations

### Example 1: Delayed Mode Smoothing
```python
files = ["sample/recording1.json", "sample/recording2.json"]
labels = ["1_person", "2_person"]
selected_method_name = "Delayed(mode)"

apply_smoothing_analysis(
    files=files,
    labels=labels,
    selected_method=selected_method_name,
    fps=4,
    output_dir="./results_delayed",
    create_visualizations=True
)
```

### Example 2: Median Filter
```python
files = ["sample/data.json"]
labels = ["3_person"]
selected_method_name = "Median(w=5)"

apply_smoothing_analysis(
    files=files,
    labels=labels,
    selected_method=selected_method_name,
    fps=4,
    output_dir="./results_median",
    create_visualizations=True
)
```

### Example 3: Combined Method
```python
files = ["sample/test1.json", "sample/test2.json", "sample/test3.json"]
labels = ["0_person", "1_person", "2_person"]
selected_method_name = "Combined(w=5,α=0.3)"

apply_smoothing_analysis(
    files=files,
    labels=labels,
    selected_method=selected_method_name,
    fps=4,
    output_dir="./results_combined",
    create_visualizations=True
)
```