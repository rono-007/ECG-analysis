import streamlit as st
import pandas as pd
import numpy as np
import os

# 1. PAGE CONFIG
st.set_page_config(page_title="Detailed Model Scores", layout="wide")

st.title("🎯 Detailed Class-Wise Performance")
st.markdown("---")

# 2. DATA LOADING
def get_detailed_scores():
    csv_path = os.path.join("data", "model_confusion_matrix.csv")
    if not os.path.exists(csv_path):
        return None
    
    # Load Confusion Matrix
    df_cm = pd.read_csv(csv_path, index_col=0)
    cm = df_cm.values
    classes = df_cm.index.tolist()
    
    rows = []
    for i, name in enumerate(classes):
        # True Positives: Diagonal element
        tp = cm[i, i]
        # False Positives: Sum of column i (excluding diagonal)
        fp = cm[:, i].sum() - tp
        # False Negatives: Sum of row i (excluding diagonal)
        fn = cm[i, :].sum() - tp
        
        # Calculate Scores
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        rows.append({
            "Condition": name,
            "Precision (PPV)": f"{precision:.2f}",
            "Recall (Sensitivity)": f"{recall:.2f}",
            "F1-Score": f"{f1:.2f}",
            "True Positives": int(tp),
            "Samples": int(tp + fn)
        })
    
    return pd.DataFrame(rows)

# 3. DISPLAY UI
scores_df = get_detailed_scores()

if scores_df is not None:
    st.write("### 📊 Per-Class Diagnostic Scorecard")
    st.info("The metrics below are calculated directly from your validated confusion matrix.")

    # Highlighting the Best and Worst performing classes
    best_class = scores_df.loc[scores_df['F1-Score'].astype(float).idxmax()]
    worst_class = scores_df.loc[scores_df['F1-Score'].astype(float).idxmin()]

    c1, c2 = st.columns(2)
    with c1:
        st.success(f"**Highest Reliability:** {best_class['Condition']} (F1: {best_class['F1-Score']})")
    with c2:
        st.warning(f"**Least Reliability:** {worst_class['Condition']} (F1: {worst_class['F1-Score']})")

    st.divider()

    # Display the full detailed table
    st.table(scores_df.set_index("Condition"))

    # Technical Explanation for Thesis
    with st.expander("📚 Metric Definitions for Clinical AI"):
        st.markdown("""
        * **Precision (PPV):** Of all patients the AI labeled as having this condition, how many actually had it?
        * **Recall (Sensitivity):** Of all patients who truly have this condition, how many did the AI successfully catch?
        * **F1-Score:** The harmonic mean of Precision and Recall. It is the most robust metric for unbalanced medical data.
        """)
        
    

else:
    st.error("⚠️ Confusion Matrix CSV not found in `/data` folder.")
    st.write("Please run your `generate_metrics.py` script first.")