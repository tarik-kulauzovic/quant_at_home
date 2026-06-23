"""Event-driven backtest for the GC failed-breakdown reclaim strategy."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy import detect_pattern


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop: float
    target: float
    risk: float
    pnl: float
    return_pct: float
    r_multiple: float
    outcome: str
    pattern_time: pd.Timestamp


def _resolve_exit(bar: pd.Series, stop: float, target: float) -> tuple[float, str]:
    """Conservative intrabar rule: stop first if both levels are touched."""
    low = float(bar["Low"])
    high = float(bar["High"])

    stop_hit = low <= stop
    target_hit = high >= target

    if stop_hit and target_hit:
        return stop, "stop"
    if stop_hit:
        return stop, "stop"
    if target_hit:
        return target, "target"
    return np.nan, ""


def run_backtest(df: pd.DataFrame) -> list[Trade]:
    trades: list[Trade] = []
    pending: dict | None = None
    active: dict | None = None

    i = 2
    while i < len(df):
        bar = df.iloc[i]
        ts = df.index[i]

        if active is not None:
            exit_price, outcome = _resolve_exit(bar, active["stop"], active["target"])
            if outcome:
                pnl = exit_price - active["entry"]
                trades.append(
                    Trade(
                        entry_time=active["entry_time"],
                        exit_time=ts,
                        entry_price=active["entry"],
                        exit_price=exit_price,
                        stop=active["stop"],
                        target=active["target"],
                        risk=active["risk"],
                        pnl=pnl,
                        return_pct=pnl / active["entry"],
                        r_multiple=pnl / active["risk"],
                        outcome=outcome,
                        pattern_time=active["pattern_time"],
                    )
                )
                active = None
            i += 1
            continue

        if pending is not None:
            if float(bar["Low"]) <= pending["stop"]:
                pending = None
            elif float(bar["High"]) >= pending["entry"]:
                active = {
                    "entry_time": ts,
                    "entry": pending["entry"],
                    "stop": pending["stop"],
                    "target": pending["target"],
                    "risk": pending["risk"],
                    "pattern_time": pending["pattern_time"],
                }
                pending = None

                exit_price, outcome = _resolve_exit(bar, active["stop"], active["target"])
                if outcome:
                    pnl = exit_price - active["entry"]
                    trades.append(
                        Trade(
                            entry_time=active["entry_time"],
                            exit_time=ts,
                            entry_price=active["entry"],
                            exit_price=exit_price,
                            stop=active["stop"],
                            target=active["target"],
                            risk=active["risk"],
                            pnl=pnl,
                            return_pct=pnl / active["entry"],
                            r_multiple=pnl / active["risk"],
                            outcome=outcome,
                            pattern_time=active["pattern_time"],
                        )
                    )
                    active = None
            i += 1
            continue

        setup = detect_pattern(df, i)
        if setup is not None:
            pending = setup

        i += 1

    return trades


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "pattern_time": t.pattern_time,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop": t.stop,
                "target": t.target,
                "risk": t.risk,
                "pnl": t.pnl,
                "return_pct": t.return_pct,
                "r_multiple": t.r_multiple,
                "outcome": t.outcome,
            }
            for t in trades
        ]
    )


def compute_summary(trades: list[Trade], initial_capital: float = 100_000.0) -> dict:
    if not trades:
        return {"total_trades": 0}

    df = trades_to_dataframe(trades)
    wins = df[df["outcome"] == "target"]
    losses = df[df["outcome"] == "stop"]

    equity = initial_capital
    equity_curve = [equity]
    for _, row in df.iterrows():
        trade_risk_dollars = equity * 0.01
        units = trade_risk_dollars / row["risk"] if row["risk"] > 0 else 0
        equity += units * row["pnl"]
        equity_curve.append(equity)

    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        dd = (peak - value) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    total_return = (equity_curve[-1] / initial_capital) - 1
    years = (df["exit_time"].iloc[-1] - df["entry_time"].iloc[0]).days / 365.25
    cagr = (equity_curve[-1] / initial_capital) ** (1 / years) - 1 if years > 0 else np.nan

    return {
        "total_trades": len(df),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(df),
        "avg_r": df["r_multiple"].mean(),
        "expectancy_r": df["r_multiple"].mean(),
        "profit_factor": wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) else np.inf,
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": max_dd * 100,
        "final_equity": equity_curve[-1],
        "equity_curve": equity_curve,
    }
