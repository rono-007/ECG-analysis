# 🏥 CardioScan AI: Neural 12-Lead ECG Interpretability Engine

[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CardioScan AI** is an advanced clinical diagnostic assistance tool that leverages deep learning to interpret 12-lead Electrocardiogram (ECG) signals. Beyond simple classification, it provides **explainable AI (XAI)** through saliency maps, highlighting the specific segments of the waveform that influenced the model's decision.

---

## 🚀 Key Features

- **Multi-Class Diagnosis**: Detects 5 major cardiac conditions:
  - `Normal`, `MI` (Myocardial Infarction), `STTC` (ST/T Change), `CD` (Conduction Disturbance), `HYP` (Hypertrophy).
- **Neural Interpretability**: Generates Grad-CAM style saliency maps to visualize anomalous waveform segments.
- **Dual-Interface Ecosystem**:
  - **Clinical Command Center**: A Streamlit-based interactive diagnostic tool for individual patient analysis.
  - **Executive Dashboard**: A FastAPI + Vanilla JS dashboard for aggregate performance metrics and validation gallery.
- **Clinical Grade Processing**: Implements Butterworth bandpass filtering (0.5Hz - 50Hz) to remove baseline drift and powerline noise.
- **High Performance**: Built on a custom 1D-ResNet architecture optimized for temporal signal processing.

---

## 🏗️ Technical Architecture

### 1. Neural Network: 1D-ResNet
The core engine is a **Residual Neural Network (ResNet)** adapted for 1D time-series data.
- **Input**: 12-channel signal (leads) with 5000 temporal samples (10 seconds @ 500Hz).
- **Architecture**:
  - 4 Residual Stages with increasing filter depth (24 → 32 → 48 → 64).
  - Large kernel sizes (k=7) to capture long-range morphological features (P-QRS-T complexes).
  - Global Average Pooling (GAP) for spatial invariance.
  - Fully Connected (FC) output layer with 5 units (logits).

### 2. Interpretability (Saliency Maps)
To bridge the gap between "black-box" AI and clinical trust, we use **Gradient-based Saliency Mapping**:
- The engine computes the gradient of the winning class's logit with respect to the input signal.
- Absolute gradients are aggregated across leads and smoothed using a 50-sample moving average window.
- The resulting "Heatmap" is overlaid on the ECG plot, marking regions of diagnostic significance.

### 3. Data Pipeline
- **Dataset**: Validated against the **PTB-XL** clinical dataset.
- **Preprocessing**: 
  - Z-Score Normalization: $\frac{x - \mu}{\sigma}$
  - Butterworth Bandpass: 0.5Hz (Low-cut) to 50Hz (High-cut).

---

## 📊 Model Evaluation & Metrics

The model is evaluated using robust clinical metrics to ensure reliability in medical contexts.

| Condition | Precision | Recall | F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 0.94 | 0.96 | 0.95 | ✅ High |
| **MI** (Infarction) | 0.88 | 0.82 | 0.85 | ⚠️ Critical |
| **STTC** | 0.85 | 0.89 | 0.87 | ✅ High |
| **CD** | 0.91 | 0.88 | 0.89 | ✅ High |
| **HYP** | 0.82 | 0.78 | 0.80 | 📉 Monitor |

> **Note**: F1-Score is prioritized due to the inherent class imbalance in cardiac pathology data.

---

## 📂 Project Structure

```text
D:\ECG_WebApp\
├── app.py                # Main Streamlit Clinical Center
├── model_arch.py         # 1D-ResNet PyTorch Implementation
├── EcgEncoderBest.pt     # Pre-trained Model Weights
├── dashboard/            # FastAPI + JS Dashboard
│   ├── backend/          # Inference Engine & API
│   └── frontend/         # Web Interface (HTML/CSS/JS)
├── data/                 # PTB-XL Subsets & Metrics
│   ├── raw_signals.npy   # Processed ECG Waveforms
│   ├── final_labels_v2.npy
│   └── model_confusion_matrix.csv
├── pages/                # Streamlit Multi-page Metrics
└── requirements.txt      # Dependency Manifest
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.12 or higher
- `pip` package manager

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/ecg-webapp.git
cd ecg-webapp

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Applications

#### **Option A: Clinical Center (Streamlit)**
Best for deep-dive diagnostic analysis.
```bash
streamlit run app.py
```

#### **Option B: Executive Dashboard (FastAPI)**
Best for high-level performance overview and API access.
```bash
python dashboard/backend/main.py
```
*The dashboard will be available at `http://127.0.0.1:8000`.*

---

## 👨‍🔬 Thesis Group Information
This project is part of a **Final Year Thesis Project** focused on Neural ECG Interpretation and Explainable AI in Cardiology.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Disclaimer: CardioScan AI is a research tool and is not intended for direct clinical diagnostic use without professional medical supervision.*
