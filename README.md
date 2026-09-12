# Stock Price Movement Predictor

**GCSRM Recruitment 2026 | Technical Track: AI & Machine Learning (Year 2) | Option A**

SPY (SPDR S&P 500 ETF Trust) next-day direction (Up/Down) classification from daily OHLCV data.

## Problem Statement

Predict the next trading day's direction (Up/Down) for a liquid equity/index using daily OHLCV data, without any
temporal data leakage, while honestly comparing the result against naive baselines under a strict chronological
evaluation.

## Objective

1. Engineer same-day ("raw") and technical-indicator ("engineered") features from daily OHLCV, using only
   information available on or before the prediction date.
2. Establish two naive baselines -- **Persistence** and **Majority Class** -- that any real model must beat to be
   considered useful.
3. Compare raw features vs raw + engineered indicators under an identical linear classifier, to test whether
   technical-indicator engineering actually helps.
4. Evaluate with a strict chronological split (no shuffling) plus a walk-forward check, and report results
   honestly -- including when the model barely beats, or fails to beat, the naive baselines.

## Dataset

- **Ticker:** SPY (SPDR S&P 500 ETF Trust). Chosen because it is one of the most liquid, heavily-traded ETFs on any
  US exchange (tens of millions of shares/day), giving a clean, continuously-traded price series free of thin-liquidity
  artifacts.
- **Source:** Yahoo Finance via the `yfinance` package -- free and scriptable, so the dataset is trivially reproducible.
- **Range:** 2005-01-03 to 2025-12-31 inclusive (5,283 trading days).
- **Columns:** `Open, High, Low, Close, Volume`, with `auto_adjust=True` (split/dividend-adjusted close).
- Downloaded once via `src/data.py::download_data()` and committed at `data/spy_daily.csv`. The notebook only ever
  *loads* this CSV (`load_data()`); it never re-downloads on a normal run.

## Methodology

### Target Definition

```
next_close = Close shifted by -1
target = 1 (Up) if next_close > Close else 0 (Down)      # ties count as Down
```

The final row (no known next-day close) is dropped **before** casting to `int`, so an undefined label can never be
silently cast into a fake `0`. This is verified inline in the notebook (Section 7-8): a `Date | Close_t | Close_t+1 |
Target` table, a manual recomputation check, and a **truncation test** -- recomputing features on data cut off at
several dates and asserting they exactly match the full-data features at those dates (proving no feature ever uses
future information).

### Features (frozen before any test-set result was seen)

**Raw (5)** -- single-day, scale-free transforms of OHLCV:

| Feature | Formula |
|---|---|
| `ret_1d` | `Close_t / Close_{t-1} - 1` |
| `range` | `(High - Low) / Close` |
| `body` | `(Close - Open) / Open` |
| `gap` | `Open_t / Close_{t-1} - 1` |
| `vol_chg` | `log(Volume_t / Volume_{t-1})` |

**Engineered = Raw + 5 technical indicators**, implemented directly in pandas (no `ta` / `pandas-ta`):

| Indicator | Captures | Implementation |
|---|---|---|
| `rsi_14` | Momentum / overbought-oversold | Wilder's RSI(14) |
| `macd_hist_norm` | Trend momentum | MACD(12,26) hist vs signal(9), normalized by Close |
| `bb_pctb` | Mean-reversion / volatility positioning | Bollinger %B, SMA20 +/- 2*std20 |
| `dist_sma50` | Medium-term trend | `Close / SMA50 - 1` |
| `vol_20` | Volatility regime | rolling 20-day std of `ret_1d` |

All windows are trailing only (`min_periods` set explicitly, never `center=True`, never `bfill`). Raw and engineered
sets are evaluated on the **same rows** -- NaNs (feature warm-up + the one undefined-target row) are dropped once,
so the comparison isn't confounded by different usable row sets.

### Baselines

- **Persistence:** predict tomorrow = today's direction (`Close_t > Close_{t-1}`, i.e. the sign of `ret_1d`).
- **Majority Class:** always predict the most common label in the *training* data only.

### Model

`StandardScaler -> LogisticRegression(C=1.0, max_iter=1000)`, identical pipeline for both feature sets, **no
hyperparameter tuning**. The scaler is part of the pipeline and is fit only on training data.

### Second Model: Random Forest (web-version requirement)

The assignment's web version adds a requirement the PDF omits: "Two models compared on the final feature set."
The PDF's four-way comparison (below) already covers one model across two feature sets plus two baselines; to
additionally satisfy the web version, a second classifier is trained on the same engineered feature set as the
Logistic Regression model:

`StandardScaler -> RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1)`, no hyperparameter
tuning. Trees don't need feature scaling, but the classifier is kept inside a `Pipeline` for symmetry with the
other approaches. `n_jobs=1` (not `-1`): parallel tree aggregation in `predict_proba` is not bit-reproducible
across runs even with `random_state` set, since worker completion order affects floating-point summation order;
the dataset is small enough that single-threaded fitting is fast. This was added after the primary experiment was
frozen, and is reported as a supplementary comparison -- it does not replace or alter any of the four required
rows in the comparison table below.

Result, reported as-is:

| Approach | Accuracy | Balanced Accuracy | % Predicted Up |
|---|---:|---:|---:|
| Engineered Features (LR) | 54.82% | 50.91% | 98.28% |
| Engineered (Random Forest) | 54.73% | 52.87% | 73.16% |

On the single split, Random Forest comes within one call of Logistic Regression (573/1,047 vs 574/1,047 correct),
while predicting Up far less often and reaching a somewhat higher balanced accuracy. In the walk-forward check,
Random Forest performs worse than Logistic Regression (mean accuracy 51.7% vs LR's 54.8% and Majority Class's
55.3%; see `outputs/walk_forward_results.csv`). Given the walk-forward gap and that the single-split near-tie sits
inside the ~3-point-wide 95% confidence intervals, Random Forest is not presented as a headline improvement -- if
anything it generalizes slightly worse across folds than the linear model.

### Train/Test Split

- Strict chronological 80/20 split on the cleaned, feature-complete dataset (5,233 usable rows) -- no shuffling.
- **1-row embargo:** the last raw-training row is dropped, since its label depends on the first test day's close
  (would otherwise leak test-period price information into a training label).
- Train: 4,185 rows, 2005-03-15 to 2021-10-26. Test: 1,047 rows, 2021-10-28 to 2025-12-30.

### Leakage Prevention (summary)

- All features use only `shift(+k)` (past data); only the target uses `shift(-1)`.
- Target constructed with drop-before-cast (see above).
- Chronological split, 1-row embargo, `StandardScaler` fit on training data only (verified: fitted `scaler.mean_`
  matches the train-only feature mean and differs from the full-dataset mean).
- Truncation test confirms every feature at date `d` is identical whether computed on the full series or on data
  cut off at `d`.

All of the above is shown and asserted inline in `notebooks/stock_prediction.ipynb`.

## Results

### Four-way comparison (single chronological 80/20 split, test set, n=1,047)

| Approach | Accuracy | 95% CI | Balanced Accuracy | % Predicted Up |
|---|---:|---:|---:|---:|
| Persistence | 51.00% | [47.97%, 54.03%] | 50.67% | 54.15% |
| Majority Class | 54.06% | [51.04%, 57.08%] | 50.00% | 100.00% |
| Raw Features | 54.35% | [51.33%, 57.36%] | 50.39% | 98.76% |
| Engineered Features | 54.82% | [51.81%, 57.84%] | 50.91% | 98.28% |

(Full table with confusion-matrix counts: `outputs/comparison_table.csv`.)

### Walk-forward validation (generalization check)

The assignment's "verify generalization under cross-session real-world constraints" is interpreted here as a
**walk-forward validation**: `TimeSeriesSplit(n_splits=5, gap=1)`, expanding training window, `gap=1` embargoing
one row per fold boundary, and **Majority Class recomputed independently within each fold's training data**. Mean
+/- std accuracy across the 5 folds:

| Approach | Mean Accuracy | Std |
|---|---:|---:|
| Majority Class | 55.32% | 0.61% |
| Engineered Features | 54.79% | 1.37% |
| Raw Features | 54.70% | 1.55% |
| Persistence | 49.63% | 1.45% |

(Full per-fold table: `outputs/walk_forward_results.csv`.)

### Class Balance

| Partition | Up | Down | n | % Up | % Down |
|---|---:|---:|---:|---:|---:|
| Full | 2,886 | 2,347 | 5,233 | 55.15% | 44.85% |
| Train | 2,319 | 1,866 | 4,185 | 55.41% | 44.59% |
| Test | 566 | 481 | 1,047 | 54.06% | 45.94% |

(`outputs/class_balance.csv`, `outputs/class_balance.png`.)

### Prediction Visualization

`outputs/predictions.png` (SPY close / actual vs predicted direction raster / rolling 60-day accuracy vs majority)
and `outputs/predictions_zoom.png` (last ~60 test days, zoomed).

## Analysis

All four approaches cluster tightly in the 49-55% range on the held-out test set, with heavily overlapping 95%
confidence intervals -- there is no statistically clear winner from the single split alone. Persistence is
consistently the weakest approach (~51% single-split, ~49.6% walk-forward mean): SPY's daily direction does not
meaningfully autocorrelate day-to-day. Majority Class is a genuinely strong baseline (~54-55%) simply because the
data has more Up days than Down days (~55% base rate); any real model needs to clear this bar to add value. The
engineered feature set edges out raw features by a small margin (54.8% vs 54.3% single-split accuracy, and
essentially tied in the walk-forward check), but balanced accuracy for both stays barely above 50%, meaning neither
model separates the two classes much better than chance once the class imbalance is accounted for. Neither logistic
model decisively beats Majority Class in either evaluation.

**Honest conclusion:** with this feature set and a simple linear model, technical-indicator engineering does not
produce a practically or statistically significant improvement over a naive base-rate baseline for next-day SPY
direction. This is consistent with the efficient-market expectation that a liquid, heavily-traded index shouldn't
have easily exploitable linear structure in daily OHLCV-derived features. Nothing here is claimed as a profitable
trading strategy.

## How to Run

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

# Data is already committed at data/spy_daily.csv, so this step is optional (re-downloads and overwrites it):
python -m src.data

jupyter nbconvert --to notebook --execute --inplace notebooks/stock_prediction.ipynb
# or open notebooks/stock_prediction.ipynb in Jupyter and Restart & Run All
```

## Requirements

See `requirements.txt` (pinned to the versions actually used). Developed with **Python 3.12.3**, the interpreter
that produced the notebook's committed outputs (see `language_info` in `notebooks/stock_prediction.ipynb`).

```
pandas==2.3.3
numpy==2.2.6
scikit-learn==1.7.2
matplotlib==3.10.9
yfinance==1.7.0
jupyter==1.1.1
```

## Project Structure

```
Stock-ml/
├── data/
│   └── spy_daily.csv              # committed SPY OHLCV, 2005-01-03 to 2025-12-31
├── notebooks/
│   └── stock_prediction.ipynb     # full pipeline + narrative + leakage checks, committed with outputs
├── src/
│   ├── data.py                    # download_data(), load_data()
│   ├── features.py                # build_target(), build_raw_features(), build_indicators(), build_dataset()
│   ├── models.py                  # persistence, majority, shared LR pipeline, and the RF pipeline
│   │                              #   (make_rf_pipeline() / fit_predict_rf() -- web-version 2nd model)
│   └── evaluation.py              # chronological split, metrics, walk-forward, required plots
├── outputs/
│   ├── comparison_table.csv
│   ├── walk_forward_results.csv
│   ├── class_balance.csv / class_balance.png
│   ├── predictions.png
│   └── predictions_zoom.png
├── requirements.txt
└── README.md
```

## Limitations

- No transaction costs, slippage, or execution modeling -- this is a classification-accuracy study, not a trading
  strategy backtest.
- Single ticker (SPY), single asset class; results may not generalize to individual equities or other timeframes.
- Linear model only -- Logistic Regression cannot capture nonlinear feature interactions.
- Small, fixed feature set (5 raw + 5 indicators), frozen deliberately to avoid overfitting via feature search.
- Walk-forward uses only 5 folds.
- No probability calibration or decision-threshold analysis; predictions use the default 0.5 threshold.
- `auto_adjust=True` dividend/split adjustment slightly changes historical price levels over the 21-year window;
  negligible here since all features are same-day ratios/differences, not raw price levels.

## Conclusion

This project builds a leakage-verified pipeline for next-day SPY direction prediction: causal, past-only features;
a target constructed and independently cross-checked to avoid a subtle NaN-casting bug; a strict chronological
80/20 split with a 1-row embargo; a scaler fit only on training data; and both single-split and walk-forward
evaluation across four approaches. The honest result is that technical-indicator feature engineering provides at
most a marginal, not clearly significant, improvement over raw features and the majority-class baseline. The value
of this project is in the rigor of the pipeline -- explicit leakage checks, naive baselines, and chronological +
walk-forward evaluation methodology -- over a headline accuracy number.
