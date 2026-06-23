"""Download COMEX gold (GC) futures and build 4-hour OHLC bars."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    CHUNK_DAYS,
    DATA_DIR,
    DATA_MODE,
    INTERVAL_SOURCE,
    RESAMPLE_RULE,
    START_DATE,
    TICKER,
)


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df.sort_index()


def download_hourly_recent() -> pd.DataFrame:
    """Download the maximum recent 1h history Yahoo allows (~730 days)."""
    raw = yf.download(
        TICKER,
        period="730d",
        interval=INTERVAL_SOURCE,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError(f"No recent 1h data returned for {TICKER}.")
    return _normalize_ohlc(raw)


def download_daily(start: str = START_DATE) -> pd.DataFrame:
    raw = yf.download(
        TICKER,
        start=start,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError(f"No daily data returned for {TICKER}.")
    return _normalize_ohlc(raw)


def synthesize_4h_from_daily(daily: pd.DataFrame, bars_per_day: int = 6) -> pd.DataFrame:
    """
    Build deterministic 4h bars from daily OHLC when intraday history is unavailable.

    Uses a zig-zag path from open to close that touches the daily high and low,
    producing realistic wicks for pattern detection.
    """
    rows: list[dict] = []

    for date, day in daily.iterrows():
        o = float(day["Open"])
        h = float(day["High"])
        l = float(day["Low"])
        c = float(day["Close"])
        vol = float(day["Volume"]) / bars_per_day

        if c >= o:
            pivot_prices = [o, l, h, c]
            pivot_idx = [0, 1, 3, bars_per_day - 1]
        else:
            pivot_prices = [o, h, l, c]
            pivot_idx = [0, 1, 3, bars_per_day - 1]

        pivot_idx_arr = np.array(pivot_idx, dtype=float)
        bar_indices = np.arange(bars_per_day)
        closes = np.interp(bar_indices, pivot_idx_arr, pivot_prices)

        opens = np.empty(bars_per_day)
        opens[0] = o
        opens[1:] = closes[:-1]

        highs = np.maximum(opens, closes)
        lows = np.minimum(opens, closes)

        wick = max((h - l) * 0.12, 0.01)
        seed = int(pd.Timestamp(date).strftime("%Y%m%d"))
        for i in range(bars_per_day):
            phase = ((seed * 31 + i * 17) % 100) / 100.0
            extra_high = wick * (0.25 + 0.75 * phase)
            extra_low = wick * (0.75 - 0.5 * phase)
            highs[i] = min(h, max(highs[i], opens[i], closes[i]) + extra_high)
            lows[i] = max(l, min(lows[i], opens[i], closes[i]) - extra_low)

        base = pd.Timestamp(date).normalize()
        for i in range(bars_per_day):
            rows.append(
                {
                    "Datetime": base + pd.Timedelta(hours=4 * i),
                    "Open": opens[i],
                    "High": highs[i],
                    "Low": lows[i],
                    "Close": closes[i],
                    "Volume": vol,
                }
            )

    out = pd.DataFrame(rows).set_index("Datetime").sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    ohlc = df.resample(RESAMPLE_RULE).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    return ohlc.dropna(subset=["Open", "High", "Low", "Close"])


def build_dataset(mode: str = DATA_MODE) -> pd.DataFrame:
    if mode == "real":
        hourly = download_hourly_recent()
        four_h = resample_to_4h(hourly)
        four_h.attrs["data_source"] = "yahoo_1h_recent"
        return four_h

    if mode == "extended":
        daily = download_daily()
        four_h = synthesize_4h_from_daily(daily)
        four_h.attrs["data_source"] = "daily_synthetic_4h"
        return four_h

    if mode == "hybrid":
        daily = download_daily()
        synth = synthesize_4h_from_daily(daily)
        recent_hourly = download_hourly_recent()
        recent_4h = resample_to_4h(recent_hourly)
        cutoff = recent_4h.index.min()
        combined = pd.concat([synth[synth.index < cutoff], recent_4h])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined.attrs["data_source"] = "hybrid_synth_plus_yahoo_1h"
        return combined

    raise ValueError(f"Unknown DATA_MODE: {mode}")


def load_or_download(
    csv_path: Path | None = None,
    force_download: bool = False,
    mode: str = DATA_MODE,
) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = csv_path or (DATA_DIR / f"gc_4h_{mode}.csv")

    if path.exists() and not force_download:
        df = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
        df.attrs["data_source"] = f"cached_{mode}"
        return df.sort_index()

    four_h = build_dataset(mode)
    four_h.index.name = "Datetime"
    four_h.to_csv(path)
    return four_h


if __name__ == "__main__":
    for m in ("real", "extended", "hybrid"):
        data = load_or_download(force_download=True, mode=m)
        print(f"{m}: bars={len(data)} range={data.index.min()} -> {data.index.max()}")
