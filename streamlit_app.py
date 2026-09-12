import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="SPY Direction Prediction", layout="wide")

st.title("SPY Daily Direction Prediction")
st.caption("Comparing baseline and engineered-feature models for next-day SPY direction.")

# ---------- 1. Class balance report ----------
CLASS_BALANCE_PATH = "outputs/class_balance.csv"

if os.path.exists(CLASS_BALANCE_PATH):
    balance_df = pd.read_csv(CLASS_BALANCE_PATH, index_col="partition")
    st.subheader("Class Balance Report")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe((balance_df[["pct_up", "pct_down"]] * 100).rename(
            columns={"pct_up": "% Up", "pct_down": "% Down"}
        ))
    with col2:
        fig, ax = plt.subplots()
        (balance_df[["pct_up", "pct_down"]] * 100).rename(
            columns={"pct_up": "Up", "pct_down": "Down"}
        ).plot(kind="bar", ax=ax)
        ax.set_ylabel("% of days")
        ax.set_title("Up vs Down Day Distribution by Partition")
        st.pyplot(fig)
else:
    st.warning(f"Could not find {CLASS_BALANCE_PATH} — class balance section skipped.")

st.divider()

# ---------- 2. Comparison table (four/five-way) ----------
st.subheader("Model Comparison Table")
COMPARISON_PATH = "outputs/comparison_table.csv"  # ADJUST if different

if os.path.exists(COMPARISON_PATH):
    comp_df = pd.read_csv(COMPARISON_PATH)
    st.dataframe(comp_df, width='stretch')
else:
    st.warning(f"Could not find {COMPARISON_PATH}")

st.divider()

# ---------- 3. Two-model head-to-head (explicit requirement) ----------
st.subheader("Two-Model Comparison on Final Feature Set")
st.caption("Logistic Regression vs Random Forest, both trained on the engineered feature set.")

if os.path.exists(COMPARISON_PATH):
    two_model = comp_df[comp_df.iloc[:, 0].str.contains("Engineered", case=False, na=False)]
    if not two_model.empty:
        st.dataframe(two_model, width='stretch')
    else:
        st.info("Adjust the filter above to match your exact row labels for LR/RF.")

st.divider()

# ---------- 4. Walk-forward results ----------
st.subheader("Walk-Forward Validation Results")
WALKFORWARD_PATH = "outputs/walk_forward_results.csv"  # ADJUST if different

if os.path.exists(WALKFORWARD_PATH):
    wf_df = pd.read_csv(WALKFORWARD_PATH)
    st.dataframe(wf_df, width='stretch')
else:
    st.warning(f"Could not find {WALKFORWARD_PATH}")

st.divider()

# ---------- 5. Predicted vs Actual plot ----------
st.subheader("Predicted vs Actual")
PLOT_IMAGE_PATH = "outputs/predictions.png"

if os.path.exists(PLOT_IMAGE_PATH):
    st.image(PLOT_IMAGE_PATH, width='stretch')
else:
    st.warning(f"Could not find {PLOT_IMAGE_PATH}.")

st.divider()
st.caption("Deployed via Streamlit Community Cloud.")
