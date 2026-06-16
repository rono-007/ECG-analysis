import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from model_arch import EcgModel
from scipy.signal import butter, filtfilt
import streamlit as st
# ... other imports

st.set_page_config(page_title="CardioScan AI: Diagnostic Center", layout="wide")

# The rest of your diagnostic code stays the same...

def apply_clinical_filter(data):
    # Bandpass 0.5Hz to 50Hz removes breathing drift and powerline hum
    nyq = 0.5 * 500 # 500 is your sampling rate
    b, a = butter(1, [0.5/nyq, 50.0/nyq], btype='band')
    return filtfilt(b, a, data, axis=1)
# ==========================================
# 1. CREATIVE UI & GLASSMORPHISM CSS
# ==========================================
st.set_page_config(page_title="CardioAI: Clinical Command Center", layout="wide")

st.markdown("""
    <style>
    /* Main Background Gradient */
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }

    /* Custom Header Card */
    .header-card {
        background: rgba(255, 255, 255, 0.8);
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        backdrop-filter: blur(4px);
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Professional Metric Cards */
    div[data-testid="stMetric"] {
        background: white !important;
        border-radius: 15px !important;
        padding: 20px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
        border-bottom: 5px solid #2e7d32 !important;
    }
    
    /* Metric Text Visibility Fix */
    [data-testid="stMetricLabel"] { color: #444444 !important; font-weight: bold !important; opacity: 1 !important; }
    [data-testid="stMetricValue"] { color: #000000 !important; font-size: 1.8rem !important; }

    /* Heartbeat Pulse Animation */
    .status-indicator {
        height: 12px; width: 12px; background-color: #2e7d32;
        border-radius: 50%; display: inline-block; margin-right: 8px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(46, 125, 50, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(46, 125, 50, 0); }
        100% { transform: scale(0.9); }
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. CONSTANTS & CACHING
# ==========================================
CLASS_NAMES = ["Normal", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
THRESHOLDS = [0.5228, 0.8866, 0.7324, 0.6470, 0.8625]
CLASS_COLORS = {"Normal": "#2e7d32", "MI": "red", "STTC": "orange", "CD": "purple", "HYP": "magenta"}

@st.cache_resource
def get_model():
    m = EcgModel()
    m.load_state_dict(torch.load("EcgEncoderBest.pt", map_location='cpu'))
    m.eval()
    return m

def normalize(sig):
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

def get_saliency(model, tensor, class_idx):
    tensor.requires_grad_()
    logits = model(tensor)
    score = logits[0, class_idx]
    model.zero_grad(); score.backward()
    grads = tensor.grad.data.abs().numpy()[0]
    mask = np.max(grads, axis=0)
    mask = np.convolve(mask, np.ones(50)/50, mode='same')
    return (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

# ==========================================
# 3. SIDEBAR & DATA
# ==========================================
try:
    raw_signals = np.load("data/raw_signals.npy")
    all_labels = np.load("data/final_labels_v2.npy")
    test_offset = len(all_labels) - 50
except:
    st.error("Data missing in /data folder")
    st.stop()

with st.sidebar:
    st.markdown("### 🏥 Diagnostic Control")
    # Using session_state for the Gallery buttons to work
    if 'p_idx' not in st.session_state: st.session_state.p_idx = 0
    p_idx = st.slider("Patient Record Index", 0, 49, key="p_idx")
    
    st.divider()
    analyze_btn = st.button("🚀 Analyze Waveforms", use_container_width=True)
    st.caption("Final Year Group Thesis Project")

# ==========================================
# 4. MAIN LAYOUT
# ==========================================
st.markdown("""
    <div class="header-card">
        <h1 style='color: #1e3a8a; margin: 0;'>CARDIOSCAN AI</h1>
        <p style='color: #64748b;'>Neural 12-Lead ECG Interpretability Engine</p>
        <div><span class="status-indicator"></span><span style='color: #2e7d32; font-weight: bold;'>CORE ACTIVE</span></div>
    </div>
    """, unsafe_allow_html=True)

if analyze_btn:
    # 1. Processing
    sig = raw_signals[p_idx]
    normal_ref = raw_signals[0] # Baseline
    gt_vec = all_labels[test_offset + p_idx]
    gt_labels = [CLASS_NAMES[j] for j, v in enumerate(gt_vec) if v == 1]
    
    model = get_model()
    input_t = torch.tensor(normalize(sig)).float().unsqueeze(0)
    logits = model(input_t)
    probs = torch.sigmoid(logits / 1.2377).detach().numpy()[0]
    
    pred_idx = np.argmax(probs)
    mask = None
    if pred_idx != 0:
        mask = get_saliency(model, input_t, pred_idx)

    # 2. Results Header
    st.subheader(f"Clinical Report: Case #{test_offset + p_idx}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Detection", CLASS_NAMES[pred_idx])
    c2.metric("Confidence", f"{probs[pred_idx]:.1%}")
    c3.metric("Expert Label", gt_labels[0] if gt_labels else "Normal")

    st.divider()

    # 3. Plotting
    st.write("### 📈 Annotated 12-Lead Visualization")
    fig = plt.figure(figsize=(16, 14), facecolor='white')
    gs = gridspec.GridSpec(12, 1)
    t = np.arange(sig.shape[1]) / 500
    
    for i in range(12):
        ax = fig.add_subplot(gs[i])
        ax.fill_between(t, normal_ref[i], color='dodgerblue', alpha=0.1) # Tally
        ax.plot(t, sig[i], color='black', linewidth=1.1) # Patient
        if mask is not None: # Anomaly Mask
            ax.fill_between(t, sig[i].min(), sig[i].max(), where=(mask > 0.68), 
                            color=CLASS_COLORS[CLASS_NAMES[pred_idx]], alpha=0.3)
        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=25, fontweight='bold')
        ax.grid(color='#FFD1D1', linewidth=0.8); ax.minorticks_on()
        for s in ['top', 'right', 'bottom']: ax.spines[s].set_visible(False)
        if i < 11: ax.set_xticklabels([])

    st.pyplot(fig)
    
    with st.expander("📊 Probability Distribution"):
        st.bar_chart(pd.DataFrame({'Class': CLASS_NAMES, 'P': probs}).set_index('Class'), color="#2e7d32")

else:
    # --- VALIDATION GALLERY ---
    st.markdown("<h2 style='text-align: center;'>Model Validation Gallery</h2>", unsafe_allow_html=True)
    st.write("Each card represents a verified case from the PTB-XL dataset used to validate the ResNet-1D architecture.")
    
    cols = st.columns(5)
    # Find indices for the 5 classes manually for the gallery
    val_map = {"Normal": 0, "MI": 0, "STTC": 0, "CD": 0, "HYP": 0}
    for i in range(50):
        v = all_labels[test_offset + i]
        for idx, name in enumerate(CLASS_NAMES):
            if v[idx] == 1: val_map[name] = i
            
    for i, name in enumerate(CLASS_NAMES):
        with cols[i]:
            c = CLASS_COLORS[name]
            st.markdown(f"""
                <div style="background: white; padding: 20px; border-radius: 12px; border-top: 6px solid {c}; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
                    <h3 style="color: {c}; margin:0;">{name}</h3>
                    <p style="color: #666; font-size: 0.9rem;">Dataset Case Index: {val_map[name]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button(f"Analyze {name}", key=f"btn_{name}"):
                st.session_state.p_idx = val_map[name]
                st.rerun()

    st.image("https://i.imgur.com/8Q8Ym7Y.png", use_container_width=True)