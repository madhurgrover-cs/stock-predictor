"""Chronological split, metrics, walk-forward validation, and required plots.

See PLAN.md > Split, > Metrics, > Outputs.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

from src.models import fit_predict, fit_predict_rf, majority_predict, persistence_predict

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def chronological_split(X: pd.DataFrame, y: pd.Series, test_frac: float = 0.2):
    """80/20 chronological split with a 1-row embargo (drop the last training row,
    since its label depends on the first test day's close).
    """
    assert X.index.equals(y.index), "X and y must share the same index"
    n = len(y)
    n_test = int(round(n * test_frac))
    n_train_raw = n - n_test

    train_idx = X.index[: n_train_raw - 1]  # embargo: drop last raw-training row
    test_idx = X.index[n_train_raw:]

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    assert X_train.index.max() < X_test.index.min(), "Train must strictly precede test"
    return X_train, X_test, y_train, y_test


def accuracy_ci95(y_true: np.ndarray, y_pred: np.ndarray):
    """Normal-approximation 95% CI for accuracy."""
    n = len(y_true)
    acc = float((y_true == y_pred).mean())
    se = np.sqrt(acc * (1 - acc) / n)
    z = 1.959963984540054
    return acc, acc - z * se, acc + z * se


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true_arr, y_pred_arr = y_true.values, y_pred.values
    acc, ci_low, ci_high = accuracy_ci95(y_true_arr, y_pred_arr)
    bal_acc = balanced_accuracy_score(y_true_arr, y_pred_arr)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    pct_up_pred = float((y_pred_arr == 1).mean())

    return {
        "accuracy": acc,
        "accuracy_ci_low": ci_low,
        "accuracy_ci_high": ci_high,
        "balanced_accuracy": bal_acc,
        "pct_predicted_up": pct_up_pred,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": len(y_true_arr),
    }


def run_comparison(X_raw, X_eng, y) -> pd.DataFrame:
    """Fit/evaluate all 4 approaches on the single chronological 80/20 split."""
    X_raw_train, X_raw_test, y_train, y_test = chronological_split(X_raw, y)
    X_eng_train, X_eng_test, _, _ = chronological_split(X_eng, y)
    assert X_raw_test.index.equals(X_eng_test.index)

    results = {}

    pred_persist = persistence_predict(X_raw_test)
    results["Persistence"] = compute_metrics(y_test, pred_persist)

    pred_majority = majority_predict(y_train, y_test.index)
    results["Majority Class"] = compute_metrics(y_test, pred_majority)

    pred_raw, _ = fit_predict(X_raw_train, y_train, X_raw_test)
    results["Raw Features"] = compute_metrics(y_test, pred_raw)

    pred_eng, _ = fit_predict(X_eng_train, y_train, X_eng_test)
    results["Engineered Features"] = compute_metrics(y_test, pred_eng)

    table = pd.DataFrame(results).T
    table.index.name = "approach"

    predictions = pd.DataFrame({
        "y_true": y_test,
        "persistence": pred_persist,
        "majority": pred_majority,
        "raw": pred_raw,
        "engineered": pred_eng,
    })

    return table, predictions, (X_raw_train, X_raw_test, X_eng_train, X_eng_test, y_train, y_test)


def walk_forward_evaluate(X_raw, X_eng, y, n_splits=5, gap=1) -> pd.DataFrame:
    """TimeSeriesSplit(n_splits, gap), expanding window, all 5 approaches per fold.
    Majority is recomputed per fold from that fold's training labels. Includes the
    Random Forest classifier on the engineered feature set as a 5th approach.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    rows = []

    for fold, (train_pos, test_pos) in enumerate(tscv.split(X_eng), start=1):
        train_idx, test_idx = X_eng.index[train_pos], X_eng.index[test_pos]

        Xr_train, Xr_test = X_raw.loc[train_idx], X_raw.loc[test_idx]
        Xe_train, Xe_test = X_eng.loc[train_idx], X_eng.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

        assert train_idx.max() < test_idx.min()

        preds = {
            "Persistence": persistence_predict(Xr_test),
            "Majority Class": majority_predict(y_train, y_test.index),
            "Raw Features": fit_predict(Xr_train, y_train, Xr_test)[0],
            "Engineered Features": fit_predict(Xe_train, y_train, Xe_test)[0],
            "Engineered (Random Forest)": fit_predict_rf(Xe_train, y_train, Xe_test)[0],
        }

        for approach, pred in preds.items():
            acc = float((pred.values == y_test.values).mean())
            rows.append({
                "fold": fold,
                "approach": approach,
                "accuracy": acc,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            })

    return pd.DataFrame(rows)


def class_balance_report(y_full, y_train, y_test) -> pd.DataFrame:
    def counts(y, label):
        up = int((y == 1).sum())
        down = int((y == 0).sum())
        n = len(y)
        return {"partition": label, "up": up, "down": down, "n": n,
                "pct_up": up / n, "pct_down": down / n}

    return pd.DataFrame([
        counts(y_full, "full"),
        counts(y_train, "train"),
        counts(y_test, "test"),
    ]).set_index("partition")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_class_balance(balance_df: pd.DataFrame, path: Path = None):
    path = path or OUTPUTS_DIR / "class_balance.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(balance_df))
    width = 0.35
    ax.bar(x - width / 2, balance_df["pct_up"] * 100, width, label="Up", color="#2ca02c")
    ax.bar(x + width / 2, balance_df["pct_down"] * 100, width, label="Down", color="#d62728")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(balance_df.index, fontsize=11)
    ax.set_ylabel("Percent of samples (%)")
    ax.set_title("Class Balance: Up vs Down by Partition")
    ax.legend()
    for i, row in enumerate(balance_df.itertuples()):
        ax.text(i - width / 2, row.pct_up * 100 + 1, f"{row.pct_up*100:.1f}%", ha="center", fontsize=9)
        ax.text(i + width / 2, row.pct_down * 100 + 1, f"{row.pct_down*100:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_predictions(df_close: pd.Series, predictions: pd.DataFrame, path: Path = None):
    """3 stacked panels over the test window:
    (1) SPY close, (2) actual vs predicted direction raster (engineered model),
    (3) rolling 60-day accuracy of engineered vs majority, with a 50% line.
    """
    path = path or OUTPUTS_DIR / "predictions.png"
    test_close = df_close.loc[predictions.index]

    correct_eng = (predictions["engineered"] == predictions["y_true"]).astype(int)
    correct_maj = (predictions["majority"] == predictions["y_true"]).astype(int)
    roll_acc_eng = correct_eng.rolling(60, min_periods=60).mean()
    roll_acc_maj = correct_maj.rolling(60, min_periods=60).mean()

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1, 1.5]})

    axes[0].plot(test_close.index, test_close.values, color="#1f77b4")
    axes[0].set_title("SPY Close Price (Test Window)")
    axes[0].set_ylabel("Close ($)")
    axes[0].set_xlabel("Date")

    raster = np.vstack([predictions["y_true"].values, predictions["engineered"].values])
    x_start, x_end = mdates.date2num(predictions.index[0]), mdates.date2num(predictions.index[-1])
    axes[1].imshow(raster, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                    extent=[x_start, x_end, 0, 2], interpolation="none")
    axes[1].set_yticks([0.5, 1.5])
    axes[1].set_yticklabels(["Predicted", "Actual"])
    axes[1].set_title("Actual vs Predicted Direction (Engineered Model; green=Up, red=Down)")
    axes[1].set_xlabel("Date")

    axes[2].plot(predictions.index, roll_acc_eng.values, label="Engineered", color="#1f77b4")
    axes[2].plot(predictions.index, roll_acc_maj.values, label="Majority", color="#7f7f7f", linestyle="--")
    axes[2].axhline(0.5, color="black", linestyle=":", linewidth=1, label="50% (chance)")
    axes[2].set_ylabel("Rolling 60-day accuracy")
    axes[2].set_xlabel("Date")
    axes[2].set_title("Rolling 60-Day Accuracy: Engineered Model vs Majority Baseline")
    axes[2].legend(loc="upper right")

    axes[0].set_xlim(predictions.index[0], predictions.index[-1])
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", labelbottom=True, labelrotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_predictions_zoom(predictions: pd.DataFrame, n_days: int = 60, path: Path = None):
    """Actual vs predicted (engineered model) for the last ~n_days test days."""
    path = path or OUTPUTS_DIR / "predictions_zoom.png"
    zoom = predictions.tail(n_days)

    n_up_pred = int((zoom["engineered"] == 1).sum())

    fig, ax = plt.subplots(figsize=(13, 3.5))
    raster = np.vstack([zoom["y_true"].values, zoom["engineered"].values])
    ax.imshow(raster, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
              extent=[0, len(zoom), 0, 2], interpolation="none")
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Predicted", "Actual"])
    ax.set_title(
        f"Actual vs Predicted Direction — Last {len(zoom)} Test Days (Engineered Model; "
        f"predicted Up on {n_up_pred}/{len(zoom)} days)"
    )
    tick_pos = np.arange(len(zoom))
    step = max(1, len(zoom) // 12)
    ax.set_xticks(tick_pos[::step])
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in zoom.index[::step]], rotation=30, ha="right")
    ax.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
