import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import os
from typing import Dict, Any, Optional

st.set_page_config(layout="wide")
# st.title("📡 TI Radar Data Visualization (Ceiling-Mounted)")
st.title("📡 IWR6843AOPEVM Data Visualization (Ceiling-Mounted)")

# ------------------ Radar & Visualization Configuration ------------------
RADAR_CONFIG = {
    "radar_height": 2.7,                  # Meters (Assumed, as it's not in the config)
    "num_range_bins": 256,                # From profileCfg numAdcSamples
    "num_doppler_bins": 32,               # From frameCfg numLoops
    "num_rx_antennas": 4,                 # From channelCfg rxChannelEn=15 (binary 1111 -> 4 antennas)
    "num_tx_antennas": 3,                 # From channelCfg txChannelEn=7 (binary 0111 -> 3 antennas)
    "range_resolution": 0.044,            # Meters/bin
    "velocity_resolution": 1.26,          # m/s/bin
    # Derived values
    "num_virtual_antennas": 12,           # 4 RX * 3 TX
    # For Az/El split view, based on a common 3-TX antenna config
    "num_azimuth_antennas": 8,
    "num_elevation_antennas": 4,
}

PLOT_XZ_LIMIT = np.ceil(RADAR_CONFIG["radar_height"] * np.tan(np.deg2rad(60)))
PLOT_Y_LIMIT = RADAR_CONFIG["radar_height"] + 0.1

# ------------------ Helper Functions ------------------
@st.cache_data
def load_and_parse_data(file_path: str) -> Optional[Dict[str, Any]]:
    """Loads and parses the multi-frame JSON data from the specified file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None

def format_heatmap(data: np.ndarray, db: bool = True) -> np.ndarray:
    """Converts heatmap data to dB scale if requested, avoiding log(0) errors."""
    if db:
        data = np.abs(data)
        return 20 * np.log10(data + 1e-12)
    else:
        return np.abs(data)

# ------------------ Load Data ------------------
# DATA_FILE = "replay_1_straightline.json"
DATA_FILE = "replay_1_randomwalk.json"
if not os.path.exists(DATA_FILE):
    st.error(f"Data file not found. Please make sure '{DATA_FILE}' is in the same directory.")
    st.stop()

radar_data_all_frames = load_and_parse_data(DATA_FILE)
if radar_data_all_frames is None or "data" not in radar_data_all_frames:
    st.error("Invalid data format in the JSON file. Could not find top-level 'data' key.")
    st.stop()

all_frames_data = radar_data_all_frames["data"]
num_frames = len(all_frames_data)

# ------------------ Sidebar & Frame Selection ------------------
st.sidebar.header("Display Options")
frame_idx = st.sidebar.slider("Select Frame", 0, num_frames - 1, 0)
use_db_scale = st.sidebar.checkbox("Use dB scale for heatmaps", value=True)
color_by = st.sidebar.radio("Color Point Cloud By:", ["None", "Velocity", "Intensity"], key="point_color")

current_frame_data = all_frames_data[frame_idx].get('frameData', {})
frame_num = current_frame_data.get('frameNum', frame_idx + 1)
num_points = current_frame_data.get('numDetectedPoints', 0)

st.sidebar.info(f"""
**Frame:** `{frame_num}`\n
**Detected Points:** `{num_points}`
""")


# ------------------ 0. Summary Plot & Stats ------------------
st.header("📈 Point Cloud Summary (All Frames)")
summary_data = {
    "Frame": [frame['frameData'].get('frameNum', i+1) for i, frame in enumerate(all_frames_data)],
    "Detected Points": [frame['frameData'].get('numDetectedPoints', 0) for i, frame in enumerate(all_frames_data)]
}
df_summary = pd.DataFrame(summary_data)

# Calculate and display stats
stats = df_summary["Detected Points"].describe()
stat_cols = st.columns(5)
stat_cols[0].metric("Mean", f"{stats['mean']:.2f}")
stat_cols[1].metric("Median", f"{df_summary['Detected Points'].median():.2f}")
stat_cols[2].metric("Std Dev", f"{stats['std']:.2f}")
stat_cols[3].metric("Min", f"{int(stats['min'])}")
stat_cols[4].metric("Max", f"{int(stats['max'])}")

fig_summary = px.scatter(df_summary, x="Frame", y="Detected Points", title="Detected Points per Frame")
fig_summary.add_vline(x=frame_num, line_width=2, line_dash="dash", line_color="red", annotation_text="Current Frame")
st.plotly_chart(fig_summary, use_container_width=True)

st.divider()

# ------------------ 1. Point Cloud Visualization ------------------
st.header(f"1. Point Cloud (Frame {frame_num})")
pc_raw = np.array(current_frame_data.get("pointCloud", []))

if pc_raw.size > 0:
    df = pd.DataFrame(pc_raw[:, :5], columns=["x", "y", "z", "velocity", "intensity"]).astype(float)
    df["y"] = RADAR_CONFIG["radar_height"] - df["y"] # Invert Y for ceiling-mount

    # --- Point Cloud Stats for Current Frame ---
    with st.expander("Show Point Cloud Statistics for Current Frame"):
        st.dataframe(df[['x', 'y', 'z']].describe(), use_container_width=True)

    # --- Plotting ---
    color_opts = {}
    if color_by == "Velocity":
        color_opts = {"color": "velocity", "color_continuous_scale": px.colors.diverging.RdBu_r, "labels": {"color": "Velocity (m/s)"}}
    elif color_by == "Intensity":
        color_opts = {"color": "intensity", "color_continuous_scale": px.colors.sequential.Viridis, "labels": {"color": "Intensity"}}

    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        st.subheader("Top-down View (X-Z)")
        fig = px.scatter(df, x="x", y="z", title="Floor Plan", **color_opts)
        fig.update_layout(
            xaxis=dict(range=[-PLOT_XZ_LIMIT, PLOT_XZ_LIMIT]),
            yaxis=dict(range=[-PLOT_XZ_LIMIT, PLOT_XZ_LIMIT], scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Side View (X-Height)")
        fig = px.scatter(df, x="x", y="y", title="Front Elevation", **color_opts)
        fig.update_layout(
            xaxis=dict(range=[-PLOT_XZ_LIMIT, PLOT_XZ_LIMIT]),
            yaxis=dict(range=[0, PLOT_Y_LIMIT]),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.subheader("3D Room View")
        fig = px.scatter_3d(df, x="x", y="z", z="y", title="3D Perspective", **color_opts)
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X (m)", range=[-PLOT_XZ_LIMIT, PLOT_XZ_LIMIT]),
                yaxis=dict(title="Z (m)", range=[-PLOT_XZ_LIMIT, PLOT_XZ_LIMIT]),
                zaxis=dict(title="Height (m)", range=[0, PLOT_Y_LIMIT]),
                aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.5)
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No point cloud data available for this frame.")

st.divider()

# ------------------ 2. Range & Noise Profile ------------------
st.header("2. Range & Noise Profile")
rp = np.array(current_frame_data.get("rangeProfile", []))
np_profile = np.array(current_frame_data.get("noiseProfile", []))

if rp.size > 0 and np_profile.size > 0:
    range_axis = np.arange(rp.size) * RADAR_CONFIG["range_resolution"]
    df_profiles = pd.DataFrame({
        'Range (m)': range_axis,
        'Signal': format_heatmap(rp, use_db_scale),
        'Noise': format_heatmap(np_profile, use_db_scale)
    })
    fig = px.line(df_profiles, x='Range (m)', y=['Signal', 'Noise'],
                  labels={'value': 'Intensity (dB)' if use_db_scale else 'Magnitude', 'variable': 'Profile'})
    fig.update_layout(title="Range and Noise Profile", legend_title="")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No Range or Noise Profile data available for this frame.")

st.divider()

# ------------------ 3. Heatmaps ------------------
st.header("3. Processed Heatmaps")
rd_raw = np.array(current_frame_data.get("rangeDopplerHeatmap", []))
az_raw = np.array(current_frame_data.get("azimuthElevationStaticHeatmap", []))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Range-Doppler")
    if rd_raw.size > 0:
        num_range_bins_rd = rd_raw.size // RADAR_CONFIG["num_doppler_bins"]
        rd_matrix = rd_raw.reshape(num_range_bins_rd, RADAR_CONFIG["num_doppler_bins"])
        rd_shifted = np.fft.fftshift(rd_matrix, axes=1)
        img = format_heatmap(rd_shifted, use_db_scale)
        range_axis = np.arange(num_range_bins_rd) * RADAR_CONFIG["range_resolution"]
        vel_axis = (np.arange(RADAR_CONFIG["num_doppler_bins"]) - RADAR_CONFIG["num_doppler_bins"] / 2) * RADAR_CONFIG["velocity_resolution"]
        fig = px.imshow(img, x=vel_axis, y=range_axis, origin="lower",
                        aspect="auto", color_continuous_scale='jet',
                        labels={'x': 'Velocity (m/s)', 'y': 'Range (m)', 'color': 'Intensity'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No Range-Doppler data.")

with col2:
    st.subheader("Range-Azimuth")
    if az_raw.size > 0:
        az_complex = az_raw[0::2] + 1j * az_raw[1::2]
        range_az_matrix = az_complex.reshape(RADAR_CONFIG["num_range_bins"], RADAR_CONFIG["num_virtual_antennas"])
        angle_fft_size = 64
        angle_fft = np.fft.fft(range_az_matrix, angle_fft_size, axis=1)
        angle_fft_shifted = np.fft.fftshift(angle_fft, axes=(1,))
        img = format_heatmap(angle_fft_shifted, use_db_scale)
        range_axis_az = np.arange(RADAR_CONFIG["num_range_bins"]) * RADAR_CONFIG["range_resolution"]
        angles_deg = np.rad2deg(np.arcsin(np.linspace(-1, 1, angle_fft_size)))
        fig = px.imshow(img, x=angles_deg, y=range_axis_az, origin="lower",
                        aspect="auto", color_continuous_scale='jet',
                        labels={'x': 'Angle (°)', 'y': 'Range (m)', 'color': 'Intensity'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No Azimuth-Elevation data.")

# --- Az-El Heatmap (Centered Below) ---
_, mid_col, _ = st.columns([1, 2, 1])
with mid_col:
    st.subheader("Azimuth-Elevation (Top-Down)")
    if 'range_az_matrix' in locals() and range_az_matrix.size > 0:
        az_va = range_az_matrix[:, :RADAR_CONFIG["num_azimuth_antennas"]]
        el_va = range_az_matrix[:, RADAR_CONFIG["num_azimuth_antennas"]:]
        
        angle_fft_az = np.fft.fftshift(np.fft.fft(az_va, 64, axis=1), axes=1)
        angle_fft_el = np.fft.fftshift(np.fft.fft(el_va, 64, axis=1), axes=1)
        
        az_spectrum = np.sum(np.abs(angle_fft_az), axis=0, keepdims=True)
        el_spectrum = np.sum(np.abs(angle_fft_el), axis=0, keepdims=True)
        az_el_img = az_spectrum.T @ el_spectrum

        img_formatted = format_heatmap(az_el_img, use_db_scale)
        az_angles = np.rad2deg(np.arcsin(np.linspace(-1, 1, 64)))
        el_angles = np.rad2deg(np.arcsin(np.linspace(-1, 1, 64)))

        fig = px.imshow(img_formatted, x=az_angles, y=el_angles,
                        color_continuous_scale='jet', aspect="equal",
                        labels={'x': 'Azimuth Angle (°)', 'y': 'Elevation Angle (°)', 'color': 'Intensity'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Data not available to generate Az-El map.")