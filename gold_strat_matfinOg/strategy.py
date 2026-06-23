"""Three-candle failed-breakdown reclaim pattern (long only)."""

import pandas as pd


def is_bearish(open_: float, close: float) -> bool:
    return close < open_


def is_bullish(open_: float, close: float) -> bool:
    return close > open_


def detect_pattern(df: pd.DataFrame, idx: int) -> dict | None:
    """
    Candle 1 (idx-2): reference low only.
    Candle 2 (idx-1): bearish, breaks below candle 1 low.
    Candle 3 (idx): bullish, closes back above candle 1 low.
    """
    if idx < 2:
        return None

    c1 = df.iloc[idx - 2]
    c2 = df.iloc[idx - 1]
    c3 = df.iloc[idx]

    ref_low = float(c1["Low"])

    if not is_bearish(c2["Open"], c2["Close"]):
        return None
    if float(c2["Low"]) >= ref_low:
        return None
    if not is_bullish(c3["Open"], c3["Close"]):
        return None
    if float(c3["Close"]) <= ref_low:
        return None

    entry = float(c3["High"])
    stop = min(float(c2["Low"]), float(c3["Low"]))
    risk = entry - stop

    if risk <= 0:
        return None

    target = entry + risk

    return {
        "pattern_time": df.index[idx],
        "ref_low": ref_low,
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk": risk,
        "reward_r": 1.0,
    }
