import os
import time
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.fft import fft, fftfreq
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="AstroGuard Ultimate",
    page_icon=":satellite_antenna:",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --ag-bg-1: #050b17;
            --ag-bg-2: #0a1222;
            --ag-panel: #0f1a31;
            --ag-card: #0d172c;
            --ag-text: #e6edf8;
            --ag-muted: #9db1ce;
            --ag-border: #263655;
            --ag-ok: #22c55e;
            --ag-warn: #f59e0b;
            --ag-crit: #ef4444;
        }

        .stApp {
            background:
                radial-gradient(1000px 550px at 8% -12%, #18345f 0%, transparent 58%),
                linear-gradient(180deg, var(--ag-bg-1) 0%, var(--ag-bg-2) 100%);
            color: var(--ag-text);
        }

        [data-testid="stHeader"] {
            background: transparent !important;
            border-bottom: none !important;
        }

        [data-testid="stDecoration"] {
            display: none !important;
        }

        [data-testid="stToolbar"],
        [data-testid="stStatusWidget"],
        #MainMenu,
        button[title="View options"],
        button[title="Manage app"],
        button[title="Settings"] {
            display: none !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1529 0%, #0a1222 100%);
            border-right: 1px solid var(--ag-border);
        }

        [data-testid="stSidebar"] * {
            color: var(--ag-text) !important;
        }

        h1, h2, h3, p, label, span {
            color: var(--ag-text) !important;
        }

        [data-testid="stCaptionContainer"] p {
            color: var(--ag-muted) !important;
            letter-spacing: 0.03em;
        }

        div[data-baseweb="input"] input {
            background-color: var(--ag-panel) !important;
            color: var(--ag-text) !important;
            border: 1px solid var(--ag-border) !important;
        }

        button[kind] {
            background: #13203b !important;
            border: 1px solid var(--ag-border) !important;
            color: var(--ag-text) !important;
            white-space: nowrap !important;
            font-weight: 600 !important;
            min-height: 2.6rem !important;
        }

        .metric-card {
            background: linear-gradient(180deg, rgba(15, 26, 49, 0.97) 0%, rgba(10, 18, 34, 0.97) 100%);
            border: 1px solid var(--ag-border);
            border-left: 6px solid #64748b;
            border-radius: 10px;
            margin-bottom: 10px;
            padding: 14px;
            color: var(--ag-text);
            line-height: 1.5;
        }

        .time-display {
            font-family: Consolas, monospace;
            font-size: 32px;
            color: #00ff85;
            background: #020617;
            border: 1px solid var(--ag-border);
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        }

        .rul-display {
            font-size: 48px;
            font-weight: 800;
            text-align: center;
            border: 1px solid var(--ag-border);
            border-radius: 10px;
            padding: 14px;
            background: rgba(2, 6, 23, 0.8);
        }

        .status-ok { border-left-color: var(--ag-ok) !important; }
        .status-warn { border-left-color: var(--ag-warn) !important; }
        .status-crit { border-left-color: var(--ag-crit) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AstroGuard: Predictive Maintenance Suite")
st.caption("Demo mode: NASA IMS bearing dataset replay (not live spacecraft telemetry).")

st.sidebar.header("Control Panel")
# ИЗМЕНЕНО: Относительный путь, который будет работать внутри Docker
default_path = "./IMS"


def browse_data_folder(initial_dir: str) -> str:
    # ИЗМЕНЕНО: Добавлена защита (try-except) для работы без графического интерфейса в Docker
    try:
        import tkinter as tk
        from tkinter import filedialog

        # Если мы в Linux без монитора, tkinter упадет здесь
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        try:
            selected = filedialog.askdirectory(
                initialdir=initial_dir if os.path.isdir(initial_dir) else os.getcwd(),
                title="Select telemetry data folder",
            )
        finally:
            root.destroy()
        return selected
    except Exception:
        # Вместо краша выдаем понятную ошибку
        raise Exception("Кнопка недоступна в Docker. Введите путь вручную (например, ./IMS)")


if "data_path" not in st.session_state:
    st.session_state["data_path"] = default_path
if "browse_error" not in st.session_state:
    st.session_state["browse_error"] = ""


def on_browse_click() -> None:
    try:
        picked_path = browse_data_folder(st.session_state.get("data_path", default_path))
        if picked_path:
            st.session_state["data_path"] = picked_path
        st.session_state["browse_error"] = ""
    except Exception as exc:
        st.session_state["browse_error"] = str(exc)


st.sidebar.text_input("Data Source:", key="data_path")
st.sidebar.button("Browse Folder", use_container_width=True, on_click=on_browse_click)
st.sidebar.caption("Expected format: folder with many IMS files named YYYY.MM.DD.HH.MM.SS")
if st.session_state.get("browse_error"):
    st.sidebar.error(f"Folder picker failed: {st.session_state['browse_error']}")

data_path = st.session_state["data_path"]
speed = st.sidebar.slider("Simulation Speed (ms)", 10, 500, 50)
window_size = st.sidebar.slider("Smoothing Window", 5, 50, 20)
start_btn = st.sidebar.button("> INITIATE MISSION")


def parse_timestamp(filename: str) -> str:
    try:
        parts = filename.split(".")
        dt_str = ".".join(parts[:6])
        dt_obj = datetime.strptime(dt_str, "%Y.%m.%d.%H.%M.%S")
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return "UNKNOWN TIME"


def compute_fft(signal: np.ndarray, sample_rate: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    sample_count = len(signal)
    yf = fft(signal)
    xf = fftfreq(sample_count, 1 / sample_rate)
    idx = xf >= 0
    return xf[idx], 2.0 / sample_count * np.abs(yf[idx])


@st.cache_resource(show_spinner=False)
def init_system(path: str):
    files = sorted([f for f in os.listdir(path) if not f.startswith(".")])
    if len(files) < 320:
        raise ValueError(f"Need at least 320 telemetry files; found {len(files)}.")

    train_files = files[:300]
    data_list = []

    for filename in train_files[::2]:
        try:
            df = pd.read_csv(os.path.join(path, filename), sep="\t", header=None)
            rms = np.sqrt((df ** 2).mean())
            data_list.append(rms.values)
        except Exception:
            continue

    if not data_list:
        raise ValueError(
            "Could not parse training data from selected folder. Expected tab-separated numeric IMS files."
        )

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(data_list)

    model = MLPRegressor(hidden_layer_sizes=(8, 4), activation="tanh", max_iter=500)
    model.fit(x_train, x_train)

    reconstructions = model.predict(x_train)
    mse = np.mean(np.square(x_train - reconstructions), axis=1)
    threshold = float(np.max(mse) * 2.0)

    return model, scaler, files, threshold


if start_btn:
    if not os.path.exists(data_path):
        st.sidebar.error("Data source path does not exist.")
        st.stop()

    status_text = st.sidebar.empty()
    status_text.info("Booting AI Core...")

    try:
        model, scaler, all_files, threshold = init_system(data_path)
    except Exception as exc:
        st.sidebar.error(f"Startup failed: {exc}")
        st.stop()

    status_text.success("System Online")

    col_time, col_rul = st.columns([2, 1])
    with col_time:
        st.caption("ONBOARD TIME (dataset timestamp)")
        time_placeholder = st.empty()
    with col_rul:
        st.caption("AI PREDICTION (RUL, heuristic)")
        rul_placeholder = st.empty()

    st.divider()

    col_main, col_info = st.columns([3, 1])
    with col_main:
        st.subheader("HEALTH TREND (anomaly score vs alert limit)")
        st.caption("Green line is anomaly score. Red dashed line is alert limit from training baseline.")
        chart_placeholder = st.empty()
        st.subheader("SPECTRAL SIGNATURE (FFT of vibration data)")
        st.caption("Shows where vibration energy is concentrated by frequency.")
        spectrogram_placeholder = st.empty()

    with col_info:
        st.subheader("WHAT THIS MEANS (30 sec)")
        st.markdown(
            """
            <div class="metric-card">
                <b>1)</b> Data comes from NASA IMS bearing tests, replayed from files.<br>
                <b>2)</b> The model learns a baseline from early files, then scores new files by difference.<br>
                <b>3)</b> Trend chart: green = anomaly score, red = alert limit.<br>
                <b>4)</b> RUL is a heuristic trend extrapolation, not guaranteed failure time.<br>
                <b>5)</b> Time is parsed from file names, so dates can show 2004.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("DIAGNOSTIC LOG (rule-based)")
        log_placeholder = st.empty()

    trend_history = deque(maxlen=window_size)
    plot_data: list[float] = []

    start_index = 300

    for i in range(start_index, len(all_files), 5):
        filename = all_files[i]
        filepath = os.path.join(data_path, filename)

        try:
            current_time = parse_timestamp(filename)
            time_placeholder.markdown(
                f'<div class="time-display">{current_time}</div>',
                unsafe_allow_html=True,
            )

            df_raw = pd.read_csv(filepath, sep="\t", header=None)
            signal = df_raw[0].values

            rms_current = np.sqrt((df_raw ** 2).mean()).values.reshape(1, -1)
            rms_scaled = scaler.transform(rms_current)
            reconstruction = model.predict(rms_scaled)
            instant_loss = float(np.mean(np.square(rms_scaled - reconstruction)))

            trend_history.append(instant_loss)
            current_trend = float(sum(trend_history) / len(trend_history))
            plot_data.append(current_trend)

            rul_text = "CALCULATING..."
            rul_color = "#94a3b8"

            if len(plot_data) > 10:
                growth = current_trend - plot_data[-10]
                if growth > 0.0001:
                    steps_left = (threshold * 3 - current_trend) / (growth / 10)
                    hours_left = max((steps_left * 10) / 60, 0)

                    if hours_left > 240:
                        rul_text = "> 10 DAYS"
                        rul_color = "#22c55e"
                    else:
                        rul_text = f"{hours_left:.1f} HOURS"
                        rul_color = "#ef4444" if hours_left < 24 else "#f59e0b"
                else:
                    rul_text = "STABLE"
                    rul_color = "#22c55e"

            rul_placeholder.markdown(
                f"<div class='rul-display' style='color: {rul_color};'>{rul_text}</div>",
                unsafe_allow_html=True,
            )

            if current_trend < threshold:
                status_msg = f"""
                <div class="metric-card status-ok">
                    <b>STATUS: NOMINAL</b><br>
                    Current score: {current_trend:.4f}<br>
                    Alert limit: {threshold:.4f}<br>
                    Rule: score &lt; limit
                </div>
                """
            elif current_trend < threshold * 2:
                status_msg = f"""
                <div class="metric-card status-warn">
                    <b>STATUS: ADVISORY</b><br>
                    Current score: {current_trend:.4f}<br>
                    Alert limit: {threshold:.4f}<br>
                    Rule: limit &lt;= score &lt; 2 x limit
                </div>
                """
            else:
                status_msg = f"""
                <div class="metric-card status-crit">
                    <b>STATUS: CRITICAL</b><br>
                    Current score: {current_trend:.4f}<br>
                    Alert limit: {threshold:.4f}<br>
                    Rule: score >= 2 x limit
                </div>
                """

            log_placeholder.markdown(status_msg, unsafe_allow_html=True)

            fig_trend, ax_trend = plt.subplots(figsize=(10, 3), dpi=120)
            fig_trend.patch.set_facecolor("#070f1f")
            ax_trend.set_facecolor("#0b172f")
            ax_trend.plot(plot_data, color="#22c55e", lw=2.1, label="Anomaly Score")
            ax_trend.axhline(y=threshold, color="#ef4444", linestyle="--", lw=1.4, label="Limit")
            ax_trend.set_xlabel("File step (replay index)", color="#cbd5e1", fontsize=9)
            ax_trend.set_ylabel("Anomaly score (reconstruction error)", color="#cbd5e1", fontsize=9)

            y_max = max(max(plot_data), threshold) * 1.15
            ax_trend.set_ylim(0, y_max if y_max > 0 else 1)

            ax_trend.tick_params(colors="#cbd5e1", labelsize=9)
            for spine in ax_trend.spines.values():
                spine.set_color("#334155")
            ax_trend.grid(color="#1e293b", linestyle="--", linewidth=0.6, alpha=0.7)
            ax_trend.legend(loc="upper left", frameon=False, fontsize=8, labelcolor="#e2e8f0")
            chart_placeholder.pyplot(fig_trend, use_container_width=True)
            plt.close(fig_trend)

            if i % 10 == 0:
                freqs, amps = compute_fft(signal)
                fig, ax = plt.subplots(figsize=(8, 2.5), dpi=120)
                fig.patch.set_facecolor("#070f1f")
                ax.set_facecolor("#0b172f")
                ax.plot(freqs, amps, color="#38bdf8", lw=0.9)

                freq_limit = min(5000, float(freqs[-1])) if len(freqs) else 5000
                ax.set_xlim(0, freq_limit)
                if len(amps):
                    y_spectrum_max = max(0.2, float(np.max(amps)) * 1.1)
                else:
                    y_spectrum_max = 0.2
                ax.set_ylim(0, y_spectrum_max)


                ax.set_xlabel("Frequency (Hz)", color="#cbd5e1", fontsize=8)
                ax.set_ylabel("Amplitude", color="#cbd5e1", fontsize=8)
                ax.tick_params(colors="#cbd5e1", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color("#334155")
                ax.grid(color="#1e293b", linestyle="--", linewidth=0.5, alpha=0.7)
                spectrogram_placeholder.pyplot(fig, use_container_width=True)
                plt.close(fig)

            time.sleep(speed / 1000)

        except Exception as exc:
            log_placeholder.error(f"Processing halted near '{filename}': {exc}")
            break
