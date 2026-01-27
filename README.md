# 🛰️ AstroGuard: AI-Driven Spacecraft Predictive Maintenance

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![ML](https://img.shields.io/badge/AI-Autoencoder-green)

**AstroGuard** is an Edge AI predictive maintenance system designed to detect early-stage anomalies in spacecraft reaction wheels. By analyzing vibration telemetry in real-time, it predicts mechanical failures days before they become critical, preventing mission loss (e.g., Kepler telescope scenario).

## 🚀 Key Features

* **Early Anomaly Detection:** Uses an **Autoencoder Neural Network (Unsupervised Learning)** to identify deviations from the "healthy" baseline without needing historical failure data.
* **RUL Prediction:** Calculates **Remaining Useful Life** in hours, allowing engineers to switch to redundant systems safely.
* **Sensor Fusion:** Combines AI-driven trend analysis with **FFT (Fast Fourier Transform)** spectral analysis to verify physical defects.
* **Edge-Optimized:** Built with `scikit-learn` for lightweight performance, ready for deployment on onboard satellite computers (OBC) via Docker.
* **Mission Control Dashboard:** A "Houston-style" interface for real-time monitoring of onboard time, health status, and spectral signatures.

## 🛠️ Tech Stack

* **Core:** Python 3.10
* **Machine Learning:** Scikit-learn (MLPRegressor / Autoencoder)
* **Signal Processing:** SciPy (FFT, Signal analysis)
* **Visualization & UI:** Streamlit, Matplotlib
* **Data Processing:** Pandas, NumPy

## 📊 Data Source

The model was trained and validated using the **NASA IMS Bearing Dataset** (PCoE).
* **Experiment:** Rexnord ZA-2115 bearings subjected to 6000 lbs load at 2000 RPM until failure (35 days test).
* **Source:** [NASA Open Data Portal](https://data.nasa.gov/dataset/ims-bearings) provided by the Center for Intelligent Maintenance Systems (IMS), University of Cincinnati.

## ⚙️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/lsimg/AstroGuard.git](https://github.com/lsimg/AstroGuard.git)
    cd AstroGuard
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Simulation:**
    ```bash
    python -m streamlit run app.py
    ```

---
**License:** MIT