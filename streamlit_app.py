"""Interactive Streamlit app for the SPY next-day direction project.

Presentation layer only. All computation is delegated to src/ (the same
functions the notebook uses) so nothing here can drift from the graded
pipeline: src.data.load_data, src.features.build_dataset /
build_raw_features / build_indicators, src.models.*, src.evaluation.*.
"""

import io

import numpy as np
import pandas as pd
import streamlit as st

from src.data import TICKER, load_data
from src.evaluation import (
    chronological_split,
    class_balance_report,
    compute_metrics,
    plot_class_balance,
    plot_predictions,
    plot_predictions_zoom,
    run_comparison,
    walk_forward_evaluate,
)
from src.features import (
    ENGINEERED_FEATURE_COLUMNS,
    INDICATOR_COLUMNS,
    RAW_FEATURE_COLUMNS,
    build_dataset,
    build_indicators,
    build_raw_features,
)
from src.models import fit_predict, fit_predict_rf, majority_predict, persistence_predict

st.set_page_config(page_title="SPY Direction Prediction", layout="wide")

ACCENT = "#3ddc97"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    f"""
<style>
h1, h2, h3, .section-header h2 {{
    font-family: Georgia, 'Times New Roman', serif;
}}
.section-header {{
    border-left: 4px solid {ACCENT};
    padding-left: 0.8rem;
    margin: 0.4rem 0 1.2rem 0;
}}
.section-header h2 {{
    margin: 0;
    font-size: 1.7rem;
}}
.section-header .subtitle {{
    color: #9aa0a6;
    font-size: 0.92rem;
    margin-top: 0.2rem;
}}
.metric-card {{
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    background: rgba(255,255,255,0.035);
    height: 100%;
}}
.metric-card .m-label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #9aa0a6;
}}
.metric-card .m-value {{
    font-size: 2rem;
    font-weight: 700;
    color: {ACCENT};
    line-height: 1.25;
}}
.metric-card .m-sub {{
    font-size: 0.78rem;
    color: #b0b4ba;
    margin-top: 0.15rem;
}}
.indicator-card {{
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 10px;
    padding: 1rem 1.15rem;
    background: rgba(255,255,255,0.035);
    margin-bottom: 0.9rem;
}}
.indicator-card .i-name {{
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 1.15rem;
    color: {ACCENT};
    margin-bottom: 0.35rem;
}}
.indicator-card code {{
    background: rgba(255,255,255,0.09);
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
}}
.indicator-card .i-desc {{
    margin-top: 0.5rem;
    color: #c7cad0;
    font-size: 0.92rem;
}}
.persistent-note {{
    border: 1px solid rgba(61,220,151,0.35);
    border-left: 4px solid {ACCENT};
    border-radius: 6px;
    padding: 0.55rem 0.9rem;
    background: rgba(61,220,151,0.07);
    font-size: 0.85rem;
    margin-bottom: 0.55rem;
    color: #d8dade;
}}
.kv-row {{
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.09);
}}
.kv-row .kv-label {{ color: #9aa0a6; }}
.kv-row .kv-value {{ color: #f5f5f0; font-weight: 600; }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, sub: str = None) -> None:
    sub_html = f'<div class="m-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="metric-card"><div class="m-label">{label}</div>'
        f'<div class="m-value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="section-header"><h2>{title}</h2>'
        f'<div class="subtitle">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def indicator_card(name: str, formula: str, description: str) -> None:
    st.markdown(
        f'<div class="indicator-card"><div class="i-name">{name}</div>'
        f'<code>{formula}</code><div class="i-desc">{description}</div></div>',
        unsafe_allow_html=True,
    )


def kv_row(label: str, value) -> None:
    st.markdown(
        f'<div class="kv-row"><span class="kv-label">{label}</span>'
        f'<span class="kv-value">{value}</span></div>',
        unsafe_allow_html=True,
    )


def persistent_notes() -> None:
    st.markdown(
        '<div class="persistent-note">The last training row is always dropped '
        '(1-row embargo) and the <code>StandardScaler</code> is always fit on '
        'training data only — in every configuration on this page.</div>'
        '<div class="persistent-note">Across every configuration tested here, '
        'results land close to the majority-class base rate. This is a '
        'methodology demonstration, not a trading signal.</div>',
        unsafe_allow_html=True,
    )


def fig_to_buf(save_fn, *args, **kwargs) -> io.BytesIO:
    """Call one of src.evaluation's plot_* functions against an in-memory buffer
    instead of a filesystem path, so the app never writes to disk."""
    buf = io.BytesIO()
    save_fn(*args, path=buf, **kwargs)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Cached data / computation (all delegated to src/)
# ---------------------------------------------------------------------------

@st.cache_data
def get_raw_data() -> pd.DataFrame:
    return load_data()


@st.cache_data
def get_dataset(df: pd.DataFrame):
    return build_dataset(df)


@st.cache_data
def get_default_split(X_eng: pd.DataFrame, y: pd.Series):
    return chronological_split(X_eng, y, test_frac=0.2)


@st.cache_data
def get_comparison(X_raw: pd.DataFrame, X_eng: pd.DataFrame, y: pd.Series):
    return run_comparison(X_raw, X_eng, y)


@st.cache_data
def get_walk_forward(X_raw: pd.DataFrame, X_eng: pd.DataFrame, y: pd.Series, n_splits: int):
    return walk_forward_evaluate(X_raw, X_eng, y, n_splits=n_splits, gap=1)


@st.cache_data
def get_custom_comparison(
    X_eng: pd.DataFrame, X_raw: pd.DataFrame, y: pd.Series,
    selected_cols: tuple, test_frac: float, model_choice: str,
):
    X_custom = X_eng[list(selected_cols)]
    X_train, X_test, y_train, y_test = chronological_split(X_custom, y, test_frac=test_frac)
    X_raw_test = X_raw.loc[X_test.index]

    results = {}
    results["Persistence"] = compute_metrics(y_test, persistence_predict(X_raw_test))
    results["Majority Class"] = compute_metrics(y_test, majority_predict(y_train, y_test.index))

    if model_choice == "Logistic Regression":
        pred_model, _ = fit_predict(X_train, y_train, X_test)
        model_label = "Selected Features (Logistic Regression)"
    else:
        pred_model, _ = fit_predict_rf(X_train, y_train, X_test)
        model_label = "Selected Features (Random Forest)"
    results[model_label] = compute_metrics(y_test, pred_model)

    table = pd.DataFrame(results).T
    table.index.name = "approach"
    return table, X_train.index, X_test.index


@st.cache_data
def get_truncation_test(df: pd.DataFrame, dates: tuple):
    """Reproduce the notebook's leakage truncation test: recompute raw features
    and indicators on data cut off at each date, and confirm they exactly match
    the values computed on the full series at that same date."""
    full_raw = build_raw_features(df)
    full_ind = build_indicators(df, full_raw)

    rows = []
    for d in dates:
        trunc = df.loc[:d]
        raw_t = build_raw_features(trunc)
        ind_t = build_indicators(trunc, raw_t)

        raw_match = bool(np.allclose(raw_t.loc[d].values, full_raw.loc[d].values, equal_nan=True))
        ind_match = bool(np.allclose(ind_t.loc[d].values, full_ind.loc[d].values, equal_nan=True))
        rows.append({
            "Truncation date": d.date(),
            "Raw features match": "PASS" if raw_match else "FAIL",
            "Indicators match": "PASS" if ind_match else "FAIL",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def render_overview(df, X_raw, X_eng, y):
    section_header("Overview", "SPY next-day direction: a leak-free time-series ML methodology demo")
    persistent_notes()

    X_train, X_test, _, _ = get_default_split(X_eng, y)

    cols = st.columns(5)
    with cols[0]:
        metric_card("Total Rows", f"{len(y):,}", "after feature warm-up + target drop")
    with cols[1]:
        metric_card("Raw Features", str(len(RAW_FEATURE_COLUMNS)))
    with cols[2]:
        metric_card("Indicators", str(len(INDICATOR_COLUMNS)))
    with cols[3]:
        metric_card("Engineered Features", str(len(ENGINEERED_FEATURE_COLUMNS)), "raw + indicators")
    with cols[4]:
        metric_card("Test Observations", f"{len(X_test):,}", "default 80/20 split")

    st.write("")
    st.markdown(
        f"This app predicts the next trading day's direction (Up/Down) for **{TICKER}** "
        "from daily OHLCV data, using only information available on or before the "
        "prediction date. It exists to demonstrate a rigorous, leakage-checked "
        "evaluation methodology — causal features, naive baselines, a strict "
        "chronological split, and honest reporting — rather than to produce a "
        "trading signal. Use the sidebar to explore the dataset, methodology, "
        "indicators, and live model comparisons."
    )


def render_dataset(df, X_raw, X_eng, y):
    section_header("Dataset", "Source data and the default train/test partition")
    persistent_notes()

    col1, col2 = st.columns([1, 1])
    with col1:
        kv_row("Symbol", TICKER)
        kv_row("Full name", "SPDR S&amp;P 500 ETF Trust")
        kv_row("Source", "Yahoo Finance (yfinance)")
        kv_row("Frequency", "Daily")
        kv_row("Date range", f"{df.index.min().date()} to {df.index.max().date()}")
        kv_row("Rows (raw OHLCV)", f"{len(df):,}")
        kv_row("Labeled rows (after warm-up)", f"{len(y):,}")
        kv_row("Missing values", int(df.isna().sum().sum()))
        kv_row("Duplicate dates", int(df.index.duplicated().sum()))

    with col2:
        X_train, X_test, y_train, y_test = get_default_split(X_eng, y)
        balance = class_balance_report(y, y_train, y_test)
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Train Rows", f"{len(X_train):,}",
                        f"{X_train.index.min().date()} → {X_train.index.max().date()}")
        with c2:
            metric_card("Test Rows", f"{len(X_test):,}",
                        f"{X_test.index.min().date()} → {X_test.index.max().date()}")
        st.write("")
        c3, c4 = st.columns(2)
        with c3:
            metric_card("% Up (Train)", f"{balance.loc['train', 'pct_up'] * 100:.2f}%")
        with c4:
            metric_card("% Up (Test)", f"{balance.loc['test', 'pct_up'] * 100:.2f}%")


def render_methodology(df, X_raw, X_eng, y):
    section_header("Methodology", "Target construction, causality constraints, and the split")

    st.markdown("**Target construction**")
    st.code(
        "next_close = Close.shift(-1)\n"
        "target = 1 (Up) if next_close > Close else 0 (Down)   # ties count as Down\n\n"
        "# The final row (unknown next-day close) is dropped BEFORE casting to int,\n"
        "# so an undefined label can never be silently cast into a fake 0.",
        language="python",
    )

    st.markdown("**Key constraints**")
    st.markdown(
        "- Every feature uses only `shift(+k)` (past data); only the target uses `shift(-1)`.\n"
        "- The target is constructed with drop-before-cast, verified live in the "
        "Leakage Verification section.\n"
        "- All rolling/EWM windows are trailing only (`min_periods` set explicitly, "
        "never `center=True`, never back-filled).\n"
        "- Raw and engineered feature sets are evaluated on the exact same rows, "
        "so comparisons aren't confounded by different usable row counts."
    )

    persistent_notes()

    X_train, X_test, _, _ = get_default_split(X_eng, y)
    st.markdown("**Split and embargo**")
    st.markdown(
        f"The default chronological split allocates the first "
        f"**{len(X_train):,} rows** ({X_train.index.min().date()} to "
        f"{X_train.index.max().date()}) to training and the remaining "
        f"**{len(X_test):,} rows** ({X_test.index.min().date()} to "
        f"{X_test.index.max().date()}) to testing — no shuffling. The last raw "
        "training row is then dropped (the embargo) because its label depends on "
        "the first test day's close, which would otherwise leak test-period price "
        "information into a training label. The `StandardScaler` inside every "
        "model pipeline is fit only on the training partition."
    )


INDICATOR_INFO = [
    ("RSI(14)", "100 - 100 / (1 + avg_gain(14) / avg_loss(14))",
     "Wilder's relative strength index — momentum / overbought-oversold positioning."),
    ("MACD Histogram (normalized)", "(EMA(12) - EMA(26) - Signal(9)) / Close",
     "Trend momentum: MACD histogram vs. its signal line, scaled by price."),
    ("Bollinger %B", "(Close - (SMA20 - 2·std20)) / (4·std20)",
     "Mean-reversion / volatility positioning within the Bollinger band."),
    ("Distance from SMA50", "Close / SMA(50) - 1",
     "Medium-term trend: how far price sits from its 50-day average."),
    ("20-Day Volatility", "rolling std(ret_1d, 20)",
     "Volatility regime, measured from the rolling standard deviation of daily returns."),
]


def render_indicators(df, X_raw, X_eng, y):
    section_header("Technical Indicators", "5 causal, trailing-window indicators, implemented in pure pandas")
    st.caption("No `ta` or `pandas-ta` — every indicator is computed directly in `src/features.py`.")
    for name, formula, desc in INDICATOR_INFO:
        indicator_card(name, formula, desc)


def render_model_comparison(df, X_raw, X_eng, y):
    section_header("Model Comparison", "Build a custom feature set and compare it live against the baselines")
    persistent_notes()

    st.markdown("**Split fraction**")
    train_frac = st.slider("Train fraction", 0.6, 0.9, 0.8, 0.05, label_visibility="collapsed")
    test_frac = round(1 - train_frac, 2)

    st.markdown("**Feature set**")
    fcols = st.columns(5)
    raw_selected = []
    for i, col in enumerate(RAW_FEATURE_COLUMNS):
        with fcols[i]:
            if st.checkbox(col, value=True, key=f"raw_{col}"):
                raw_selected.append(col)
    icols = st.columns(5)
    ind_selected = []
    for i, col in enumerate(INDICATOR_COLUMNS):
        with icols[i]:
            if st.checkbox(col, value=True, key=f"ind_{col}"):
                ind_selected.append(col)

    selected = raw_selected + ind_selected
    model_choice = st.radio("Model", ["Logistic Regression", "Random Forest"], horizontal=True)

    if not selected:
        st.warning("Select at least one feature to run a live comparison.")
    else:
        table, train_idx, test_idx = get_custom_comparison(
            X_eng, X_raw, y, tuple(selected), test_frac, model_choice,
        )
        st.caption(
            f"Train: {train_idx.min().date()} → {train_idx.max().date()} "
            f"({len(train_idx):,} rows)  |  Test: {test_idx.min().date()} → "
            f"{test_idx.max().date()} ({len(test_idx):,} rows)"
        )
        display = table.copy()
        display["accuracy"] = (display["accuracy"] * 100).round(2).astype(str) + "%"
        display["balanced_accuracy"] = (display["balanced_accuracy"] * 100).round(2).astype(str) + "%"
        display["pct_predicted_up"] = (display["pct_predicted_up"] * 100).round(2).astype(str) + "%"
        st.dataframe(
            display[["accuracy", "balanced_accuracy", "pct_predicted_up", "n"]],
            width="stretch",
        )

    st.divider()
    st.markdown("**Walk-forward validation**")
    st.caption("Expanding-window `TimeSeriesSplit`, always run on the full raw and engineered feature sets.")
    n_folds = st.slider("Folds", 3, 8, 5, 1)
    wf = get_walk_forward(X_raw, X_eng, y, n_folds)
    summary = wf.groupby("approach")["accuracy"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    summary_display = (summary * 100).round(2).astype(str) + "%"
    st.dataframe(summary_display, width="stretch")
    with st.expander("Per-fold results"):
        st.dataframe(wf, width="stretch")


def render_class_balance(df, X_raw, X_eng, y):
    section_header("Class Balance", "Up vs. Down distribution across the full dataset and the default split")

    X_train, X_test, y_train, y_test = get_default_split(X_eng, y)
    balance = class_balance_report(y, y_train, y_test)

    col1, col2 = st.columns([1, 2])
    with col1:
        display = balance[["up", "down", "n", "pct_up", "pct_down"]].copy()
        display["pct_up"] = (display["pct_up"] * 100).round(2).astype(str) + "%"
        display["pct_down"] = (display["pct_down"] * 100).round(2).astype(str) + "%"
        st.dataframe(display, width="stretch")
    with col2:
        buf = fig_to_buf(plot_class_balance, balance)
        st.image(buf, width="stretch")


def render_predictions(df, X_raw, X_eng, y):
    section_header("Predictions", "Predicted vs. actual direction on the default 80/20 test window")

    _, predictions_df, _ = get_comparison(X_raw, X_eng, y)
    buf = fig_to_buf(plot_predictions, df["Close"], predictions_df)
    st.image(buf, width="stretch")

    with st.expander("Zoom: last 60 test days"):
        zoom_buf = fig_to_buf(plot_predictions_zoom, predictions_df, n_days=60)
        st.image(zoom_buf, width="stretch")


def render_leakage_verification(df, X_raw, X_eng, y):
    section_header("Leakage Verification", "Truncation test, run live: features must never see the future")
    st.markdown(
        "For each date below, features are recomputed on data cut off at that date "
        "and compared against the same features computed on the full series. A "
        "match at every date proves no feature ever used information from beyond "
        "its own row."
    )

    warmup = 70  # past every indicator's warm-up window (SMA50, MACD(26,9), etc.)
    candidate_idx = df.index[warmup:-1]
    n_dates = 5
    step = max(1, len(candidate_idx) // n_dates)
    dates = tuple(candidate_idx[::step][:n_dates])

    result = get_truncation_test(df, dates)
    st.dataframe(result, width="stretch")

    if (result[["Raw features match", "Indicators match"]] == "PASS").all(axis=None):
        st.success(f"All {len(result)} truncation checks PASS — no look-ahead leakage detected.")
    else:
        st.error("At least one truncation check FAILED.")


def render_findings(df, X_raw, X_eng, y):
    section_header("Findings & Limitations", "Honest results, computed live from the default split")
    persistent_notes()

    table, predictions_df, _ = get_comparison(X_raw, X_eng, y)
    wf = get_walk_forward(X_raw, X_eng, y, 5)
    wf_summary = wf.groupby("approach")["accuracy"].mean().sort_values(ascending=False)

    persist_acc = table.loc["Persistence", "accuracy"] * 100
    majority_acc = table.loc["Majority Class", "accuracy"] * 100
    raw_acc = table.loc["Raw Features", "accuracy"] * 100
    eng_acc = table.loc["Engineered Features", "accuracy"] * 100
    eng_bal = table.loc["Engineered Features", "balanced_accuracy"] * 100

    st.markdown(
        f"- On the single chronological 80/20 split, all four required approaches "
        f"cluster tightly: Persistence {persist_acc:.2f}%, Majority Class "
        f"{majority_acc:.2f}%, Raw Features {raw_acc:.2f}%, Engineered Features "
        f"{eng_acc:.2f}%.\n"
        f"- Engineered features edge out raw features by "
        f"{eng_acc - raw_acc:.2f} points, but balanced accuracy "
        f"({eng_bal:.2f}%) stays barely above chance — the model has mostly "
        f"learned the class base rate, not real directional signal.\n"
        f"- In the walk-forward check (5 expanding folds), "
        f"**{wf_summary.index[0]}** has the highest mean accuracy "
        f"({wf_summary.iloc[0] * 100:.2f}%); the ranking between the naive "
        f"Majority Class baseline and the trained models is close, and neither "
        f"trained model decisively beats it.\n"
        f"- Persistence (\"tomorrow = today\") is consistently the weakest "
        f"approach in both evaluations, indicating SPY's daily direction does "
        f"not meaningfully autocorrelate day-to-day."
    )

    st.markdown("**Limitations**")
    st.markdown(
        "- No transaction costs, slippage, or execution modeling — this is a "
        "classification-accuracy study, not a trading strategy backtest.\n"
        "- Single ticker, single asset class; results may not generalize elsewhere.\n"
        "- Linear model (Logistic Regression) cannot capture nonlinear feature "
        "interactions; Random Forest is reported only as a supplementary check.\n"
        "- Small, fixed feature set (5 raw + 5 indicators), frozen deliberately to "
        "avoid overfitting via feature search.\n"
        "- No probability calibration or decision-threshold analysis; predictions "
        "use the default 0.5 threshold."
    )

    st.markdown(
        "**Honest conclusion:** with this feature set and these models, "
        "technical-indicator engineering does not produce a practically or "
        "statistically significant improvement over a naive base-rate baseline "
        "for next-day SPY direction. Nothing here is claimed as a profitable "
        "trading strategy."
    )


SECTIONS = [
    ("1  Overview", render_overview),
    ("2  Dataset", render_dataset),
    ("3  Methodology", render_methodology),
    ("4  Technical Indicators", render_indicators),
    ("5  Model Comparison", render_model_comparison),
    ("6  Class Balance", render_class_balance),
    ("7  Predictions", render_predictions),
    ("8  Leakage Verification", render_leakage_verification),
    ("9  Findings & Limitations", render_findings),
]


def main():
    with st.sidebar:
        st.markdown("## SPY Direction Predictor")
        st.caption("Leak-Free Time-Series ML")
        st.divider()
        labels = [label for label, _ in SECTIONS]
        choice = st.radio("Sections", labels, label_visibility="collapsed")
        st.divider()
        st.caption("Educational · Not financial advice")

    df = get_raw_data()
    X_raw, X_eng, y = get_dataset(df)

    render_fn = dict(SECTIONS)[choice]
    render_fn(df, X_raw, X_eng, y)


main()
