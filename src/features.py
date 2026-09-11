"""Target construction and feature engineering for SPY next-day direction prediction.

See PLAN.md > Target and > Features for the exact decisions this module implements.
All features use only shift(+k) (past data). Only the target uses shift(-1).
"""

import numpy as np
import pandas as pd

RAW_FEATURE_COLUMNS = ["ret_1d", "range", "body", "gap", "vol_chg"]
INDICATOR_COLUMNS = ["rsi_14", "macd_hist_norm", "bb_pctb", "dist_sma50", "vol_20"]
ENGINEERED_FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + INDICATOR_COLUMNS


def build_target(df: pd.DataFrame) -> pd.Series:
    """1 = next day's close is higher than today's close, 0 = Down (ties count as Down).

    The final row (unknown next-day close) is dropped BEFORE casting to int, so a
    NaN next_close can never be silently cast into a fake Down (0) label.
    """
    next_close = df["Close"].shift(-1)
    valid = next_close.notna()

    n_dropped = int((~valid).sum())
    assert n_dropped == 1, f"Expected exactly 1 row with unknown next-day close, got {n_dropped}"
    assert df.index[~valid][0] == df.index[-1], "The dropped row must be the last row"

    close = df["Close"][valid]
    next_close = next_close[valid]

    target = (next_close > close).astype(int)
    target.name = "target"
    return target


def build_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single-day, scale-free transforms of OHLCV. No multi-day windows."""
    close, open_, high, low, volume = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]

    features = pd.DataFrame(index=df.index)
    features["ret_1d"] = close / close.shift(1) - 1
    features["range"] = (high - low) / close
    features["body"] = (close - open_) / open_
    features["gap"] = open_ / close.shift(1) - 1
    features["vol_chg"] = np.log(volume / volume.shift(1))
    return features


def build_indicators(df: pd.DataFrame, raw_features: pd.DataFrame) -> pd.DataFrame:
    """5 technical indicators, pure pandas, all trailing/causal windows."""
    close = df["Close"]
    indicators = pd.DataFrame(index=df.index)

    # RSI(14), Wilder's smoothing
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    indicators["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD(12,26) histogram vs signal(9), normalized by Close
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    indicators["macd_hist_norm"] = (macd - signal) / close

    # Bollinger %B (SMA20 +/- 2*std20)
    sma20 = close.rolling(window=20, min_periods=20).mean()
    std20 = close.rolling(window=20, min_periods=20).std()
    upper, lower = sma20 + 2 * std20, sma20 - 2 * std20
    indicators["bb_pctb"] = (close - lower) / (upper - lower)

    # Distance from SMA50
    sma50 = close.rolling(window=50, min_periods=50).mean()
    indicators["dist_sma50"] = close / sma50 - 1

    # 20-day rolling volatility of daily returns
    indicators["vol_20"] = raw_features["ret_1d"].rolling(window=20, min_periods=20).std()

    return indicators


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (X_raw, X_engineered, y) aligned on the SAME rows.

    NaNs (feature warm-up + the one undefined-target row) are dropped exactly once,
    from the combination of engineered features + target, so that the raw-only and
    engineered approaches are compared on an identical row set.
    """
    raw = build_raw_features(df)
    indicators = build_indicators(df, raw)
    engineered = pd.concat([raw, indicators], axis=1)
    target = build_target(df)

    combined = pd.concat([engineered, target], axis=1).dropna()

    y = combined["target"].astype(int)
    X_raw = combined[RAW_FEATURE_COLUMNS]
    X_engineered = combined[ENGINEERED_FEATURE_COLUMNS]

    return X_raw, X_engineered, y
