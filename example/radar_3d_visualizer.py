import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

# ------------------ Configuration ------------------
RADAR_CONFIG = {
    "radar_height": 2.7,  # Meters
}

PLOT_XZ_LIMIT = np.ceil(RADAR_CONFIG["radar_height"] * np.tan(np.deg2rad(60)))
PLOT_Y_LIMIT = RADAR_CONFIG["radar_height"] + 0.1

# ------------------ Load Data ------------------
DATA_FILE = "example/sample_data/binData/11_09_2025_12_17_47/replay_1.json"  # Change this to your JSON file path
COLOR_BY = "Velocity"  # Options: "None", "Velocity", "Intensity"

print("Loading data...")
with open(DATA_FILE, 'r') as f:
    radar_data_all_frames = json.load(f)

all_frames_data = radar_data_all_frames["data"]
num_frames = len(all_frames_data)
print(f"Loaded {num_frames} frames")

# ------------------ Create Figure with Slider ------------------
fig = go.Figure()

# Process all frames and add as traces
for frame_idx in range(num_frames):
    current_frame_data = all_frames_data[frame_idx].get('frameData', {})
    frame_num = current_frame_data.get('frameNum', frame_idx + 1)
    num_points = current_frame_data.get('numDetectedPoints', 0)

    pc_raw = np.array(current_frame_data.get("pointCloud", []))

    if pc_raw.size > 0:
        df = pd.DataFrame(pc_raw[:, :5], columns=["x", "y", "z", "velocity", "intensity"]).astype(float)
        df["y"] = RADAR_CONFIG["radar_height"] - df["y"]

        # Determine color
        if COLOR_BY == "Velocity":
            color_vals = df["velocity"]
            colorscale = "RdBu_r"
            colorbar_title = "Velocity (m/s)"
        elif COLOR_BY == "Intensity":
            color_vals = df["intensity"]
            colorscale = "Viridis"
            colorbar_title = "Intensity"
        else:
            color_vals = None
            colorscale = None
            colorbar_title = None

        # Add trace for this frame
        scatter = go.Scatter3d(
            x=df["x"],
            y=df["z"],
            z=df["y"],
            mode='markers',
            marker=dict(
                size=4,
                color=color_vals if color_vals is not None else 'blue',
                colorscale=colorscale,
                showscale=(frame_idx == 0),  # Show colorbar only for first trace
                colorbar=dict(title=colorbar_title) if colorbar_title else None,
                line=dict(width=0)
            ),
            name=f"Frame {frame_num}",
            visible=(frame_idx == 0),  # Only first frame visible initially
            text=[f"Point {i}<br>Vel: {df.iloc[i]['velocity']:.2f}<br>Int: {df.iloc[i]['intensity']:.2f}" 
                  for i in range(len(df))],
            hovertemplate='X: %{x:.2f}m<br>Z: %{y:.2f}m<br>Height: %{z:.2f}m<br>%{text}<extra></extra>'
        )
        fig.add_trace(scatter)
    else:
        # Add empty trace for frames with no data
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            name=f"Frame {frame_num} (no data)",
            visible=(frame_idx == 0)
        ))

# Create slider steps
steps = []
for i in range(num_frames):
    frame_num = all_frames_data[i].get('frameData', {}).get('frameNum', i + 1)
    num_points = all_frames_data[i].get('frameData', {}).get('numDetectedPoints', 0)

    step = dict(
        method="update",
        args=[{"visible": [False] * num_frames},
              {"title": f"3D Point Cloud - Frame {frame_num} ({num_points} points)"}],
        label=str(frame_num)
    )
    step["args"][0]["visible"][i] = True
    steps.append(step)

# Add slider
sliders = [dict(
    active=0,
    yanchor="top",
    y=0,
    xanchor="left",
    currentvalue=dict(
        prefix="Frame: ",
        visible=True,
        xanchor="right"
    ),
    pad=dict(b=10, t=50),
    len=0.9,
    x=0.1,
    steps=steps
)]

# Update layout
fig.update_layout(
    sliders=sliders,
    scene=dict(
        xaxis=dict(title="X (m)", range=[-PLOT_XZ_LIMIT, PLOT_XZ_LIMIT]),
        yaxis=dict(title="Z (m)", range=[-PLOT_XZ_LIMIT, PLOT_XZ_LIMIT]),
        zaxis=dict(title="Height (m)", range=[0, PLOT_Y_LIMIT]),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.5)
    ),
    title="3D Point Cloud - Frame 1",
    showlegend=False,
    width=1200,
    height=800,
    margin=dict(l=0, r=0, b=100, t=40)
)

print("Opening interactive 3D plot in browser...")
print("Use the slider at the bottom to change frames")
fig.show()
