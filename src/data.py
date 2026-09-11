"""Data download and loading for SPY daily OHLCV data.

See PLAN.md > Dataset for the exact decisions this module implements.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

TICKER = "SPY"
START = "2005-01-01"
END = "2026-01-01"  # yfinance `end` is exclusive -> covers through 2025-12-31
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "spy_daily.csv"

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Download SPY daily OHLCV from Yahoo Finance and save to `path`."""
    df = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)

    if df.empty:
        raise RuntimeError("yfinance returned no data for SPY; check ticker/date range/network.")

    # yfinance may return MultiIndex columns (Price, Ticker) even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[REQUIRED_COLUMNS].copy()
    df.index.name = "Date"
    df.columns.name = None

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)

    return df


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load SPY daily OHLCV from `path` and validate it."""
    df = pd.read_csv(path, index_col="Date", parse_dates=True)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df[REQUIRED_COLUMNS].copy()

    assert df.index.is_monotonic_increasing, "Dates are not sorted ascending"
    assert not df.index.duplicated().any(), "Duplicate dates found"
    assert not df.isna().any().any(), "NaNs found in data"
    assert (df[["Open", "High", "Low", "Close"]] > 0).all().all(), "Non-positive price found"
    assert (df["Volume"] > 0).all(), "Non-positive volume found"

    return df


if __name__ == "__main__":
    data = download_data()
    print(f"Downloaded {len(data)} rows")
    print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")
    print("\nFirst rows:")
    print(data.head())
    print("\nLast rows:")
    print(data.tail())

    validated = load_data()
    print(f"\nload_data() validation passed: {len(validated)} rows")
