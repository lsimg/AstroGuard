# AstroGuard: Edge AI–Driven Spacecraft Predictive Maintenance

**AstroGuard** is an autonomous Edge AI predictive maintenance system designed for early anomaly detection in spacecraft reaction wheels.  
By analyzing vibration telemetry in real time, the system predicts mechanical failures **7.0 ± 0.4 days before critical breakdown**, significantly outperforming classical statistical baselines.

The objective is to prevent mission-ending failures and reduce operational losses.

---

## Core Capabilities

### Early Anomaly Detection  
AstroGuard uses an **unsupervised Deep MLP Autoencoder** trained exclusively on nominal (healthy) baseline data.  
It detects deviations without requiring labeled failure datasets — critical for space missions where failure data is scarce.

### Remaining Useful Life (RUL) Forecasting  
Provides a **7-day predictive horizon**, enabling proactive mitigation strategies before critical system degradation.

### Physics-Informed Validation  
AI anomaly scores are cross-validated using **Fast Fourier Transform (FFT)** spectral analysis to confirm physical defect signatures, including:

- Ball Pass Frequency Outer (BPFO) ≈ 166 Hz  
- Harmonic frequency amplification  
- Spectral energy drift patterns  

This hybrid approach increases interpretability and reliability.

### Edge-Optimized Architecture  

Designed for onboard satellite processors:

- < 20 ms inference time  
- < 10 MB RAM footprint  
- 99.9% reduction in telemetry bandwidth  
- Operates within ~10 kbps downlink constraints  

The system is fully containerized via Docker for hardware-agnostic deployment.

### Mission Control Dashboard  

Streamlit-based monitoring interface providing:

- Real-time anomaly score (MSE)  
- Dynamic threshold tracking (μ + 3σ)  
- Spectral defect visualization  
- Health trend progression  

---

## Technology Stack

| Layer | Tools |
|-------|-------|
| Core Language | Python 3.10 |
| Machine Learning | Scikit-learn (MLP Autoencoder) |
| Signal Processing | SciPy (FFT, spectral analysis) |
| Visualization | Streamlit, Matplotlib |
| Containerization | Docker |

---

## Repository Structure

The project includes two implementations illustrating the transition from research prototype to edge-ready deployment.

### `main.py` — Research Prototype

- **Framework:** TensorFlow / Keras  
- **Purpose:** Deep learning experimentation and proof-of-concept validation  
- **Output:** Static matplotlib visualizations  

### `app.py` — Production / Edge Deployment

- **Framework:** Scikit-learn  
- **Purpose:** Optimized model for embedded satellite systems  
- **Key Optimization:**  
  - Reduced model size  
  - Sub-20 ms inference  
  - Minimal RAM footprint  
  - OBC-compatible  

---

## Dataset

Training and validation were conducted using the **NASA IMS Bearing Dataset**.

- **Test Rig:** Rexnord ZA-2115 bearings  
- **Load:** 6,000 lbs radial  
- **Speed:** 2,000 RPM  
- **Sampling:** Downsampled to 10 kHz  
- **Segmentation:** Overlapping 1-second windows  
- **Normalization:** MinMax scaling  

This dataset simulates progressive bearing degradation until failure.

---

## Installation & Execution

### Option A — Docker (Recommended)

```bash
git clone https://github.com/lsimg/AstroGuard.git
cd AstroGuard
docker build -t astroguard .
docker run -p 8501:8501 astroguard
```

Access the dashboard at:

```
http://localhost:8501
```

---

### Option B — Local Execution

```bash
git clone https://github.com/lsimg/AstroGuard.git
cd AstroGuard
pip install -r requirements.txt
streamlit run app.py
```

---

## Performance Summary

| Metric | Result |
|--------|--------|
| Failure Prediction Horizon | 7.0 ± 0.4 days |
| Inference Latency | < 20 ms |
| RAM Usage | < 10 MB |
| Telemetry Reduction | 99.9% |

---

## License

MIT License
