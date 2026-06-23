"""Charts for backtest and Monte Carlo results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT_DIR

_plot_dir = OUTPUT_DIR


def _ensure_output_dir() -> Path:
    _plot_dir.mkdir(parents=True, exist_ok=True)
    return _plot_dir


def plot_equity_curve(trades_df: pd.DataFrame, initial_capital: float, save: bool = True) -> None:
    out = _ensure_output_dir()
    equity = [initial_capital]
    dates = [trades_df["entry_time"].iloc[0]]

    capital = initial_capital
    for _, row in trades_df.iterrows():
        risk_dollars = capital * 0.01
        units = risk_dollars / row["risk"] if row["risk"] > 0 else 0
        capital += units * row["pnl"]
        equity.append(capital)
        dates.append(row["exit_time"])

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(dates, equity, linewidth=1.8, color="#1f77b4")
    ax.axhline(initial_capital, linestyle="--", color="gray", linewidth=1)
    ax.set_title("Equity Curve (1% Risk per Trade)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Account Equity ($)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(out / "01_equity_curve.png", dpi=150)
    plt.close(fig)


def plot_drawdown(trades_df: pd.DataFrame, initial_capital: float, save: bool = True) -> None:
    out = _ensure_output_dir()
    equity = [initial_capital]
    dates = [trades_df["entry_time"].iloc[0]]

    capital = initial_capital
    for _, row in trades_df.iterrows():
        risk_dollars = capital * 0.01
        units = risk_dollars / row["risk"] if row["risk"] > 0 else 0
        capital += units * row["pnl"]
        equity.append(capital)
        dates.append(row["exit_time"])

    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(dates, dd, 0, color="#d62728", alpha=0.35)
    ax.plot(dates, dd, color="#d62728", linewidth=1.2)
    ax.set_title("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(out / "02_drawdown.png", dpi=150)
    plt.close(fig)


def plot_r_multiple_distribution(trades_df: pd.DataFrame, save: bool = True) -> None:
    out = _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(trades_df["r_multiple"], bins=30, color="#2ca02c", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(trades_df["r_multiple"].mean(), color="#ff7f0e", linewidth=2, label=f"Mean: {trades_df['r_multiple'].mean():.2f}R")
    ax.set_title("R-Multiple Distribution")
    ax.set_xlabel("R-Multiple")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(out / "03_r_multiple_hist.png", dpi=150)
    plt.close(fig)


def plot_win_loss_pie(trades_df: pd.DataFrame, save: bool = True) -> None:
    out = _ensure_output_dir()
    wins = (trades_df["outcome"] == "target").sum()
    losses = (trades_df["outcome"] == "stop").sum()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        [wins, losses],
        labels=["Target (+1R)", "Stop (-1R)"],
        autopct="%1.1f%%",
        colors=["#2ca02c", "#d62728"],
        startangle=90,
    )
    ax.set_title(f"Win / Loss Split ({len(trades_df)} trades)")
    fig.tight_layout()
    if save:
        fig.savefig(out / "04_win_loss_pie.png", dpi=150)
    plt.close(fig)


def plot_trade_returns_over_time(trades_df: pd.DataFrame, save: bool = True) -> None:
    out = _ensure_output_dir()
    colors = trades_df["r_multiple"].apply(lambda r: "#2ca02c" if r > 0 else "#d62728")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(trades_df["exit_time"], trades_df["r_multiple"], color=colors, width=20, alpha=0.85)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Trade Results Over Time (R-Multiples)")
    ax.set_xlabel("Exit Date")
    ax.set_ylabel("R-Multiple")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(out / "05_trade_results_timeline.png", dpi=150)
    plt.close(fig)


def plot_monthly_heatmap(trades_df: pd.DataFrame, save: bool = True) -> None:
    out = _ensure_output_dir()
    monthly = trades_df.copy()
    monthly["year"] = monthly["exit_time"].dt.year
    monthly["month"] = monthly["exit_time"].dt.month
    pivot = monthly.groupby(["year", "month"])["r_multiple"].sum().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.45)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-5, vmax=5)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Monthly R-Multiple Heatmap")
    fig.colorbar(im, ax=ax, label="Sum of R")
    fig.tight_layout()
    if save:
        fig.savefig(out / "06_monthly_r_heatmap.png", dpi=150)
    plt.close(fig)


def plot_monte_carlo_paths(mc_results: np.ndarray, save: bool = True) -> None:
    out = _ensure_output_dir()
    x = np.arange(mc_results.shape[1])
    p5 = np.percentile(mc_results, 5, axis=0)
    p50 = np.percentile(mc_results, 50, axis=0)
    p95 = np.percentile(mc_results, 95, axis=0)

    fig, ax = plt.subplots(figsize=(13, 7))
    for curve in mc_results[:300]:
        ax.plot(x, curve, alpha=0.04, linewidth=0.8, color="#1f77b4")
    ax.fill_between(x, p5, p95, alpha=0.25, color="#1f77b4", label="5th–95th percentile")
    ax.plot(x, p50, linewidth=2.5, color="#ff7f0e", label="Median")
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_title("Monte Carlo Equity Paths (Resampled Trades)")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Equity Multiple")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(out / "07_monte_carlo_paths.png", dpi=150)
    plt.close(fig)


def plot_monte_carlo_histogram(mc_results: np.ndarray, save: bool = True) -> None:
    out = _ensure_output_dir()
    final = mc_results[:, -1]
    p5 = np.percentile(final, 5)
    p50 = np.percentile(final, 50)
    p95 = np.percentile(final, 95)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(final, bins=70, alpha=0.85, color="#9467bd", edgecolor="white")
    ax.axvline(p5, linestyle="--", linewidth=2, label=f"5%: {p5:.2f}x")
    ax.axvline(p50, linewidth=2.5, label=f"Median: {p50:.2f}x")
    ax.axvline(p95, linestyle="--", linewidth=2, label=f"95%: {p95:.2f}x")
    ax.axvline(1.0, color="black", linestyle=":", linewidth=1)
    ax.set_title("Monte Carlo Final Equity Distribution")
    ax.set_xlabel("Final Equity Multiple")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(out / "08_monte_carlo_final_distribution.png", dpi=150)
    plt.close(fig)


def plot_sample_patterns(df: pd.DataFrame, trades_df: pd.DataFrame, n: int = 6, save: bool = True) -> None:
    """Plot a few winning/losing setups around the pattern bars."""
    out = _ensure_output_dir()
    sample = trades_df.sample(min(n, len(trades_df)), random_state=42)

    fig, axes = plt.subplots(n, 1, figsize=(13, 3 * n))
    if n == 1:
        axes = [axes]

    for ax, (_, trade) in zip(axes, sample.iterrows()):
        pattern_time = trade["pattern_time"]
        idx = df.index.get_loc(pattern_time)
        window = df.iloc[max(0, idx - 5) : min(len(df), idx + 15)]

        ax.plot(window.index, window["Close"], color="#1f77b4", linewidth=1.2, label="Close")
        ax.axhline(trade["entry_price"], color="#2ca02c", linestyle="--", label="Entry (buy stop)")
        ax.axhline(trade["stop"], color="#d62728", linestyle="--", label="Stop")
        ax.axhline(trade["target"], color="#9467bd", linestyle="--", label="Target (1R)")
        ax.axvline(pattern_time, color="gray", linestyle=":", alpha=0.8)
        ax.set_title(
            f"Pattern @ {pattern_time.date()} | {trade['outcome'].upper()} | "
            f"{trade['r_multiple']:+.1f}R"
        )
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(out / "09_sample_setups.png", dpi=150)
    plt.close(fig)


def generate_all_plots(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    mc_results: np.ndarray,
    initial_capital: float,
    output_dir: Path | None = None,
) -> None:
    global _plot_dir
    _plot_dir = output_dir or OUTPUT_DIR
    plot_equity_curve(trades_df, initial_capital)
    plot_drawdown(trades_df, initial_capital)
    plot_r_multiple_distribution(trades_df)
    plot_win_loss_pie(trades_df)
    plot_trade_returns_over_time(trades_df)
    plot_monthly_heatmap(trades_df)
    plot_monte_carlo_paths(mc_results)
    plot_monte_carlo_histogram(mc_results)
    plot_sample_patterns(df, trades_df)
