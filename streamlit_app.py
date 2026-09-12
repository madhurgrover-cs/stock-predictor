import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="SPY Direction Prediction", layout="wide")

st.title("SPY Daily Direction Prediction")
st.caption("Comparing baseline and engineered-feature models for next-day SPY direction.")

# ---------- 1. Load raw data (for class balance report) ----------
DATA_PATH = "data/spy_daily.csv"  # ADJUST if your path differs

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.subheader("Class Balance Report")
    # ADJUST: replace 'target' with your actual label column name if different
    if "target" in df.columns:
        counts = df["target"].value_counts(normalize=True) * 100
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(counts.rename("% of days"))
        with col2:
            fig, ax = plt.subplots()
            counts.plot(kind="bar", ax=ax)
            ax.set_ylabel("% of days")
            ax.set_title("Up vs Down Day Distribution")
            st.pyplot(fig)
    else:
        st.info("Target column not found under 'target' — showing raw data preview instead.")
        st.dataframe(df.head())
else:
    st.warning(f"Could not find {DATA_PATH} — class balance section skipped.")

st.divider()

# ---------- 2. Comparison table (four/five-way) ----------
st.subheader("Model Comparison Table")
COMPARISON_PATH = "outputs/comparison_table.csv"  # ADJUST if different

if os.path.exists(COMPARISON_PATH):
    comp_df = pd.read_csv(COMPARISON_PATH)
    st.dataframe(comp_df, use_container_width=True)
else:
    st.warning(f"Could not find {COMPARISON_PATH}")

st.divider()

# ---------- 3. Two-model head-to-head (explicit requirement) ----------
st.subheader("Two-Model Comparison on Final Feature Set")
st.caption("Logistic Regression vs Random Forest, both trained on the engineered feature set.")

if os.path.exists(COMPARISON_PATH):
    two_model = comp_df[comp_df.iloc[:, 0].str.contains("Engineered", case=False, na=False)]
    if not two_model.empty:
        st.dataframe(two_model, use_container_width=True)
    else:
        st.info("Adjust the filter above to match your exact row labels for LR/RF.")

st.divider()

# ---------- 4. Walk-forward results ----------
st.subheader("Walk-Forward Validation Results")
WALKFORWARD_PATH = "outputs/walk_forward_results.csv"  # ADJUST if different

if os.path.exists(WALKFORWARD_PATH):
    wf_df = pd.read_csv(WALKFORWARD_PATH)
    st.dataframe(wf_df, use_container_width=True)
else:
    st.warning(f"Could not find {WALKFORWARD_PATH}")

st.divider()

# ---------- 5. Predicted vs Actual plot ----------
st.subheader("Predicted vs Actual")
# ADJUST: point this at your actual saved plot image, e.g. "outputs/prediction_plot.png"
PLOT_IMAGE_PATH = "outputs/prediction_plot.png"

if os.path.exists(PLOT_IMAGE_PATH):
    st.image(PLOT_IMAGE_PATH, use_container_width=True)
else:
    st.warning(
        f"Could not find {PLOT_IMAGE_PATH}. "
        "If your plot lives inside the notebook only, export it as a PNG "
        "(e.g. plt.savefig('outputs/prediction_plot.png')) and re-run."
    )

st.divider()
st.caption("Deployed via Streamlit Community Cloud.")
