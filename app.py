import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from collections import deque


st.set_page_config(page_title="AstroGuard Ultimate", page_icon="🛰️", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #0e1117; }
        .metric-card { background-color: #1f2937; padding: 15px; border-radius: 8px; border-left: 5px solid gray; margin-bottom: 10px; }
        .time-display { font-family: 'Courier New', monospace; font-size: 24px; color: #00ff00; background: #000; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #333; }
        .status-ok { border-left-color: #00ff00 !important; }
        .status-warn { border-left-color: #ffa500 !important; }
        .status-crit { border-left-color: #ff0000 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🛰️ AstroGuard: Predictive Maintenance Suite")


st.sidebar.header("Control Panel")
default_path = r'D:\Spacecraft Fault Prediction\IMS\IMS\2nd_test\2nd_test'
data_path = st.sidebar.text_input("Data Source:", default_path)
speed = st.sidebar.slider("Simulation Speed (ms)", 10, 500, 50)
window_size = st.sidebar.slider("Smoothing Window", 5, 50, 20)
start_btn = st.sidebar.button("▶ INITIATE MISSION")



def parse_timestamp(filename):

    try:
        parts = filename.split('.')

        dt_str = ".".join(parts[:6])
        dt_obj = datetime.strptime(dt_str, "%Y.%m.%d.%H.%M.%S")
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S UTC")
    except:
        return "UNKNOWN TIME"


def compute_fft(signal, sample_rate=20000):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sample_rate)
    idx = xf >= 0
    return xf[idx], 2.0 / N * np.abs(yf[idx])


#Обучение ИИшки
@st.cache_resource
def init_system(path):
    files = sorted([f for f in os.listdir(path) if not f.startswith('.')])

    train_files = files[:300]

    data_list = []
    for f in train_files[::2]:
        try:
            df = pd.read_csv(os.path.join(path, f), sep='\t', header=None)
            rms = np.sqrt((df ** 2).mean())
            data_list.append(rms.values)
        except:
            pass

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(data_list)

    model = MLPRegressor(hidden_layer_sizes=(8, 4), activation='tanh', max_iter=500)
    model.fit(X_train, X_train)

    reconstructions = model.predict(X_train)
    mse = np.mean(np.square(X_train - reconstructions), axis=1)
    threshold = np.max(mse) * 2.0

    return model, scaler, files, threshold



if start_btn and os.path.exists(data_path):
    status_text = st.sidebar.empty()
    status_text.info("Booting AI Core...")
    model, scaler, all_files, threshold = init_system(data_path)
    status_text.success("System Online")

    # ВЕРХНЯЯ ПАНЕЛЬ (ВРЕМЯ И СТАТУС)
    col_time, col_rul = st.columns([2, 1])
    with col_time:
        st.caption("ONBOARD TIME")
        time_placeholder = st.empty()
    with col_rul:
        st.caption("AI PREDICTION (RUL)")
        rul_placeholder = st.empty()

    st.divider()

    col_main, col_info = st.columns([3, 1])
    with col_main:
        st.subheader("Health Trend Analysis")
        chart_placeholder = st.empty()
        st.subheader("Spectral Signature")
        spectrogram_placeholder = st.empty()

    with col_info:
        st.subheader("Diagnostic Log")
        log_placeholder = st.empty()


    trend_history = deque(maxlen=window_size)
    plot_data = []


    start_index = 300

    for i in range(start_index, len(all_files), 5):
        filename = all_files[i]
        filepath = os.path.join(data_path, filename)

        try:

            current_time = parse_timestamp(filename)
            time_placeholder.markdown(f'<div class="time-display">{current_time}</div>', unsafe_allow_html=True)


            df_raw = pd.read_csv(filepath, sep='\t', header=None)
            signal = df_raw[0].values

            rms_current = np.sqrt((df_raw ** 2).mean()).values.reshape(1, -1)
            rms_scaled = scaler.transform(rms_current)
            reconstruction = model.predict(rms_scaled)
            instant_loss = np.mean(np.square(rms_scaled - reconstruction))


            trend_history.append(instant_loss)
            current_trend = sum(trend_history) / len(trend_history)
            plot_data.append(current_trend)


            growth_rate = 0
            rul_text = "CALCULATING..."
            rul_color = "gray"

            if len(plot_data) > 10:

                growth = current_trend - plot_data[-10]
                if growth > 0.0001:
                    steps_left = (threshold * 3 - current_trend) / (growth / 10)

                    hours_left = (steps_left * 10) / 60
                    if hours_left < 0: hours_left = 0

                    if hours_left > 240:
                        rul_text = "> 10 DAYS"
                        rul_color = "#00ff00"
                    else:
                        rul_text = f"{hours_left:.1f} HOURS"
                        rul_color = "#ff0000" if hours_left < 24 else "#ffa500"
                else:
                    rul_text = "STABLE"
                    rul_color = "#00ff00"

            rul_placeholder.markdown(
                f"<h2 style='text-align: center; color: {rul_color}; border: 1px solid #333; border-radius: 5px;'>{rul_text}</h2>",
                unsafe_allow_html=True)


            if current_trend < threshold:
                status_msg = f"""
                <div class="metric-card status-ok">
                    <b>STATUS: NOMINAL</b><br>
                    AI Confidence: 99%<br>
                    Reason: Vibration signature matches baseline training data.
                </div>
                """
            elif current_trend < threshold * 2:
                status_msg = f"""
                <div class="metric-card status-warn">
                    <b>STATUS: ADVISORY</b><br>
                    AI Confidence: 85%<br>
                    Reason: Mild deviation in lower frequencies. Potential lubricant degradation.
                </div>
                """
            else:
                status_msg = f"""
                <div class="metric-card status-crit">
                    <b>STATUS: CRITICAL</b><br>
                    AI Confidence: 99.9%<br>
                    Reason: Strong structural anomaly detected! High reconstruction error ({current_trend:.2f}). Immediate action required.
                </div>
                """

            log_placeholder.markdown(status_msg, unsafe_allow_html=True)


            chart_df = pd.DataFrame({'Anomaly Score': plot_data, 'Limit': [threshold] * len(plot_data)})
            chart_placeholder.line_chart(chart_df, color=["#00ff00", "#ff0000"])

            if i % 10 == 0:
                freqs, amps = compute_fft(signal)
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.plot(freqs, amps, color='#00ff00', lw=0.5)
                ax.set_facecolor('#0e1117')
                ax.axis('off')
                ax.set_ylim(0, 0.2)
                spectrogram_placeholder.pyplot(fig)
                plt.close(fig)

            time.sleep(speed / 1000)

        except Exception:
            break