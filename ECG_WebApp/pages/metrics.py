import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from model_arch import EcgModel

# 1. LOAD DATA & MODEL
print("🔄 Loading data and model...")
device = torch.device('cpu')
model = EcgModel()
model.load_state_dict(torch.load("EcgEncoderBest.pt", map_location=device))
model.eval()

# Load your full dataset (Ensure these paths are correct)
raw_signals = np.load("data/raw_signals.npy")    # Your 50 test signals
all_labels = np.load("data/final_labels_v2.npy") # The 21k labels

# Assuming your 50 signals are the LAST 50 of the dataset
test_labels = all_labels[-50:] 

# 2. RUN INFERENCE
print("🧠 Running AI Inference on test set...")
CLASS_NAMES = ["Normal", "MI", "STTC", "CD", "HYP"]
all_probs = []

with torch.no_grad():
    for i in range(len(raw_signals)):
        # Normalize
        sig = raw_signals[i]
        sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
        
        # Predict
        input_t = torch.tensor(sig_norm).float().unsqueeze(0)
        logits = model(input_t)
        probs = torch.sigmoid(logits / 1.2377) # Apply Temperature Scaling
        all_probs.append(probs.numpy()[0])

all_probs = np.array(all_probs)

# 3. APPLY THRESHOLDS
# Use the specific thresholds from your training phase
THRESHOLDS = [0.5228, 0.8866, 0.7324, 0.6470, 0.8625]
predictions = (all_probs >= THRESHOLDS).astype(int)

# 4. CALCULATE SCIENTIFIC METRICS
print("\n" + "="*30)
print("📊 FINAL PERFORMANCE REPORT")
print("="*30)

# Generate Classification Report
report = classification_report(
    test_labels, 
    predictions, 
    target_names=CLASS_NAMES,
    zero_division=0
)
print(report)

# Generate Confusion Matrix (Aggregated for Multi-label)
# We take the Argmax to see the "Primary" confusion for a 5x5 matrix
y_true_max = np.argmax(test_labels, axis=1)
y_pred_max = np.argmax(all_probs, axis=1)
cm = confusion_matrix(y_true_max, y_pred_max)

print("\n📍 Raw Confusion Matrix Data:")
print(cm)

# 5. SAVE FOR STREAMLIT
# You can save this as a CSV to load into your "Model Metrics" page
results_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
results_df.to_csv("data/model_confusion_matrix.csv")
print("\n✅ Metrics saved to data/model_confusion_matrix.csv")