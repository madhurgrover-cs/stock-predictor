# PLAN.md — Approved Stage 1 Design (Option A: Stock Price Movement Predictor)

Stage 1 (planning) is COMPLETE and APPROVED. Build from this plan.
Do not re-plan, and do not change these decisions without asking me first.

## Dataset
- Ticker: SPY (SPDR S&P 500 ETF), daily OHLCV from Yahoo Finance via `yfinance`.
- Range: 2005-01-01 to 2025-12-31 inclusive (note: yfinance `end` is exclusive, so pass `end="2026-01-01"`).
- Set `auto_adjust=True` explicitly (the default has changed across yfinance versions).
- yfinance may return MultiIndex columns even for one ticker; flatten them to Open, High, Low, Close, Volume.
- Download ONCE via `src/data.py::download_data()`, save to `data/spy_daily.csv`, commit the CSV.
- The notebook loads from the CSV (`load_data()`), never re-downloads by default.
- `load_data()` validates: sorted ascending dates, no duplicate dates, no NaNs, no zero/negative prices or volume.

## Target
- `next_close = close.shift(-1)`
- Drop the final row (unknown next-day close) BEFORE casting. Assert it is dropped.
- `target = (next_close > close).astype(int)` → 1 = Up, 0 = Down. Ties count as Down ("not Up"); document this.
- WARNING: casting before dropping turns NaN into 0 (fake Down label). Must not happen.

## Features
"Raw" = single-day, scale-free transforms of OHLCV (no multi-day windows):
- `ret_1d` = Close_t / Close_{t-1} - 1
- `range` = (High - Low) / Close
- `body` = (Close - Open) / Open
- `gap` = Open_t / Close_{t-1} - 1
- `vol_chg` = log(Volume_t / Volume_{t-1})

"Engineered" = raw + these 5 indicators, implemented in pure pandas (no `ta`):
- `rsi_14`: Wilder RSI, `ewm(alpha=1/14, adjust=False, min_periods=14)` on gains/losses
- `macd_hist_norm`: (MACD(12,26) - signal(9)) / Close, EMAs with `adjust=False`
- `bb_pctb`: Bollinger %B, SMA20 ± 2 * rolling std(20)
- `dist_sma50`: Close / SMA50 - 1
- `vol_20`: rolling 20-day std of `ret_1d`

Rules:
- Features only use `shift(+k)` (past). Only the target uses `shift(-1)`.
- Trailing windows only. Never `center=True`. Never `bfill`.
- Use `min_periods` so warm-up NaNs are explicit; drop warm-up rows.
- All four approaches are evaluated on the SAME rows (drop NaNs once, for everything).
- Feature list is FROZEN. No adding/removing features after seeing test results.

## Split
- Chronological 80/20 on the cleaned dataset. No shuffling, ever.
- 1-row embargo: drop the last training row (its label uses the first test day's close).
- Assert max(train dates) < min(test dates).

## Approaches (the 4-way comparison)
1. Persistence: predict tomorrow's direction = today's direction (Close_t > Close_{t-1}).
2. Majority class: most common label in TRAINING data only.
3. Raw: `Pipeline([StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)])` on raw features.
4. Engineered: identical pipeline on raw + indicators.
- No hyperparameter tuning. Scaler is fit on training data only (inside the Pipeline).

## Metrics
- Accuracy (headline), balanced accuracy, confusion matrix, % of predictions that are Up.
- 95% CI for accuracy (normal approximation, computed with numpy).
- Walk-forward check: `TimeSeriesSplit(n_splits=5, gap=1)`, expanding window, all 4 approaches per fold
  (majority recomputed per fold from that fold's train), report mean ± std.
- Interpret "verify generalization under cross-session constraints" as this walk-forward check; state that in the README.

## Leakage verification (shown in the notebook)
- Table: Date | Close_t | Close_t+1 | Target — first 5 rows, last 5 rows, and rows around the split boundary.
- Truncation test: recompute features on data cut off at several dates d; features at d must exactly equal the
  full-data features at d. Assert.
- Show the fitted scaler's `mean_` equals the TRAIN feature mean (not full-data mean).
- Print train/test date ranges.

## Outputs (`outputs/`)
- `comparison_table.csv` (4 approaches, test-set metrics)
- `walk_forward_results.csv`
- `class_balance.csv` + `class_balance.png` (Up/Down counts and % for full, train, test)
- `predictions.png`: 3 stacked panels over the test window —
  (1) SPY close, (2) actual vs predicted direction as a two-row Up/Down raster,
  (3) rolling 60-day accuracy of the engineered model vs majority baseline, with a 50% line
- `predictions_zoom.png`: actual vs predicted for the last ~60 test days
- All plots: date x-axis, title, axis labels, legend.

## Structure
```
stock-price-predictor/
├── data/spy_daily.csv
├── notebooks/stock_prediction.ipynb
├── src/{__init__.py, data.py, features.py, models.py, evaluation.py}
├── outputs/
├── requirements.txt
├── .gitignore
├── CLAUDE.md
├── PLAN.md
└── README.md
```
- Notebook imports from `src/` but shows target construction and leakage checks inline.
- Notebook must pass "Restart & Run All"; commit it WITH outputs.
- Section order: follow section 14 of CLAUDE.md.

## Reproducibility
- `requirements.txt` pinned to the versions actually installed (only packages we import + jupyter).
- Record Python version in README.
- Never hardcode result numbers anywhere; README numbers are copied from actual generated outputs.

## Out of scope (unless core is fully done and verified)
- Random Forest.
- Optional sidebar: LR on literal price levels to demonstrate non-stationarity (reported separately, NOT in the 4-way table).
