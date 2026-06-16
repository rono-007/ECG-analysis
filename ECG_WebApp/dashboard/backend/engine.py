"""
CardioScan AI — Inference Engine
Handles model loading, signal processing, saliency map generation,
and metric computation for the ECG dashboard backend.
"""

import sys
import os
import torch
import numpy as np
from scipy.signal import butter, filtfilt

# Add parent directory so we can import the model architecture
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from model_arch import EcgModel

# ── Constants ────────────────────────────────────────────────────────
CLASS_NAMES  = ["Normal", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES   = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
THRESHOLDS   = [0.5228, 0.8866, 0.7324, 0.6470, 0.8625]
CLASS_COLORS = {
    "Normal": "#16a34a",
    "MI":     "#dc2626",
    "STTC":   "#ea580c",
    "CD":     "#7c3aed",
    "HYP":    "#db2777",
}
TEMP_SCALE = 1.2377
SAMPLING_RATE = 500


# ── Singleton Model Loader ──────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        weights_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'EcgEncoderBest.pt')
        )
        _model = EcgModel()
        _model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        _model.eval()
        print(f"✅ Model loaded from {weights_path}")
    return _model


# ── Singleton Data Loader ───────────────────────────────────────────
_raw_signals = None
_all_labels  = None

def get_data():
    global _raw_signals, _all_labels
    if _raw_signals is None:
        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        )
        _raw_signals = np.load(os.path.join(data_dir, 'raw_signals.npy'))
        _all_labels  = np.load(os.path.join(data_dir, 'final_labels_v2.npy'))
        print(f"✅ Data loaded — signals: {_raw_signals.shape}, labels: {_all_labels.shape}")
    return _raw_signals, _all_labels


# ── Signal Processing ───────────────────────────────────────────────
def normalize(sig: np.ndarray) -> np.ndarray:
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)


def apply_clinical_filter(data: np.ndarray) -> np.ndarray:
    """Bandpass 0.5 – 50 Hz: removes breathing drift and powerline hum."""
    nyq = 0.5 * SAMPLING_RATE
    b, a = butter(1, [0.5 / nyq, 50.0 / nyq], btype='band')
    return filtfilt(b, a, data, axis=1)


# ── Saliency Map ────────────────────────────────────────────────────
def get_saliency(model, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
    tensor = tensor.clone().detach().requires_grad_(True)
    logits = model(tensor)
    score  = logits[0, class_idx]
    model.zero_grad()
    score.backward()
    grads = tensor.grad.data.abs().numpy()[0]
    mask  = np.max(grads, axis=0)
    mask  = np.convolve(mask, np.ones(50) / 50, mode='same')
    return ((mask - mask.min()) / (mask.max() - mask.min() + 1e-8)).tolist()


# ── Core Inference ──────────────────────────────────────────────────
def analyze_patient(patient_idx: int) -> dict:
    """Run full inference pipeline on a single patient record."""
    raw_signals, all_labels = get_data()
    model = get_model()

    test_offset = len(all_labels) - 50

    sig      = raw_signals[patient_idx]
    ref_sig  = raw_signals[0]  # baseline "normal" reference
    gt_vec   = all_labels[test_offset + patient_idx]
    gt_labels = [CLASS_NAMES[j] for j, v in enumerate(gt_vec) if v == 1]

    input_t = torch.tensor(normalize(sig)).float().unsqueeze(0)
    with torch.no_grad():
        logits = model(input_t)
    probs = torch.sigmoid(logits / TEMP_SCALE).numpy()[0]

    pred_idx   = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]

    # Saliency mask (only for abnormal findings)
    saliency = None
    if pred_idx != 0:
        saliency = get_saliency(model, input_t, pred_idx)

    # Downsample signals for JSON transfer (every 5th point → 1000 pts)
    step = 5
    time_axis   = (np.arange(sig.shape[1]) / SAMPLING_RATE)[::step].tolist()
    signal_data = sig[:, ::step].tolist()        # 12 × 1000
    ref_data    = ref_sig[:, ::step].tolist()     # 12 × 1000
    saliency_ds = saliency[::step] if saliency else None

    return {
        "case_id":       test_offset + patient_idx,
        "patient_idx":   patient_idx,
        "prediction":    pred_label,
        "pred_color":    CLASS_COLORS[pred_label],
        "confidence":    round(float(probs[pred_idx]) * 100, 1),
        "ground_truth":  gt_labels if gt_labels else ["Normal"],
        "probabilities": {name: round(float(p) * 100, 1) for name, p in zip(CLASS_NAMES, probs)},
        "time":          time_axis,
        "signals":       signal_data,
        "reference":     ref_data,
        "saliency":      saliency_ds,
        "lead_names":    LEAD_NAMES,
    }


# ── Gallery Data ────────────────────────────────────────────────────
def get_gallery() -> list:
    """Return one representative case index per class for the validation gallery."""
    raw_signals, all_labels = get_data()
    test_offset = len(all_labels) - 50
    val_map = {}
    for i in range(50):
        v = all_labels[test_offset + i]
        for idx, name in enumerate(CLASS_NAMES):
            if v[idx] == 1:
                val_map[name] = i
    return [
        {"class_name": name, "index": val_map.get(name, 0),
         "color": CLASS_COLORS[name]}
        for name in CLASS_NAMES
    ]


# ── Insights / Metrics ─────────────────────────────────────────────
def compute_insights() -> dict:
    """Run inference on all 50 test patients and return aggregate metrics."""
    raw_signals, all_labels = get_data()
    model = get_model()
    test_offset = len(all_labels) - 50
    test_labels = all_labels[test_offset:]

    all_probs = []
    with torch.no_grad():
        for i in range(len(raw_signals)):
            sig_norm = normalize(raw_signals[i])
            input_t  = torch.tensor(sig_norm).float().unsqueeze(0)
            logits   = model(input_t)
            probs    = torch.sigmoid(logits / TEMP_SCALE).numpy()[0]
            all_probs.append(probs)
    all_probs   = np.array(all_probs)
    predictions = (all_probs >= THRESHOLDS).astype(int)

    # Per-class metrics
    per_class = []
    for idx, name in enumerate(CLASS_NAMES):
        tp = int(np.sum((predictions[:, idx] == 1) & (test_labels[:, idx] == 1)))
        fp = int(np.sum((predictions[:, idx] == 1) & (test_labels[:, idx] == 0)))
        fn = int(np.sum((predictions[:, idx] == 0) & (test_labels[:, idx] == 1)))
        tn = int(np.sum((predictions[:, idx] == 0) & (test_labels[:, idx] == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

        per_class.append({
            "class":     name,
            "color":     CLASS_COLORS[name],
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "accuracy":  round(accuracy, 4),
        })

    # Confusion matrix (argmax-based for 5×5)
    y_true_max = np.argmax(test_labels, axis=1).tolist()
    y_pred_max = np.argmax(all_probs,   axis=1).tolist()

    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(y_true_max, y_pred_max):
        cm[t][p] += 1

    return {
        "class_names": CLASS_NAMES,
        "per_class":   per_class,
        "confusion_matrix": cm.tolist(),
        "total_samples":    50,
    }
