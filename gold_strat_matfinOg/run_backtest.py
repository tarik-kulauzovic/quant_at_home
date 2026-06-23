"""
GC Failed-Breakdown Reclaim Strategy — backtest + Monte Carlo.

Long-only, COMEX gold futures (GC), 4-hour bars.
Pattern: bearish break below reference low, bullish reclaim, buy stop at reclaim high.
Exit: stop at pattern low, target at 1R.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_engine import compute_summary, run_backtest, trades_to_dataframe
from config import DATA_MODE, INITIAL_CAPITAL, MONTE_CARLO_SIMULATIONS, OUTPUT_DIR
from data_loader import load_or_download
from monte_carlo import monte_carlo_summary, run_trade_monte_carlo
from plots import generate_all_plots


def print_summary(summary: dict, mc_stats: dict, trades_df: pd.DataFrame, data_source: str = "") -> None:
    print("\n" + "=" * 60)
    print("GC FAILED-BREAKDOWN RECLAIM — BACKTEST SUMMARY")
    if data_source:
        print(f"Data source:         {data_source}")
    print("=" * 60)
    print(f"Total trades:        {summary['total_trades']}")
    print(f"Wins / Losses:       {summary['wins']} / {summary['losses']}")
    print(f"Win rate:            {summary['win_rate'] * 100:.1f}%")
    print(f"Avg R:               {summary['avg_r']:.3f}")
    print(f"Profit factor:       {summary['profit_factor']:.2f}")
    print(f"Total return:        {summary['total_return_pct']:.1f}%")
    print(f"CAGR:                {summary['cagr_pct']:.1f}%")
    print(f"Max drawdown:        {summary['max_drawdown_pct']:.1f}%")
    print(f"Final equity:        ${summary['final_equity']:,.0f}")
    print(f"Date range:          {trades_df['entry_time'].iloc[0].date()} -> {trades_df['exit_time'].iloc[-1].date()}")

    print("\n" + "-" * 60)
    print(f"MONTE CARLO ({MONTE_CARLO_SIMULATIONS:,} simulations)")
    print("-" * 60)
    print(f"Worst outcome:       {mc_stats['worst']:.2f}x")
    print(f"5th percentile:      {mc_stats['p5']:.2f}x")
    print(f"Median:              {mc_stats['median']:.2f}x")
    print(f"95th percentile:     {mc_stats['p95']:.2f}x")
    print(f"Best outcome:        {mc_stats['best']:.2f}x")
    print(f"P(profit):           {mc_stats['prob_profit'] * 100:.1f}%")
    print(f"P(>50% drawdown):    {mc_stats['prob_ruin_50pct'] * 100:.1f}%")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest GC failed-breakdown reclaim strategy")
    parser.add_argument("--force-download", action="store_true", help="Re-download GC data")
    parser.add_argument(
        "--mode",
        choices=["real", "extended", "hybrid"],
        default=DATA_MODE,
        help="Data source: real (730d 1h), extended (daily->4h synth), hybrid",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip chart generation")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    print("Loading GC 4h data...")
    df = load_or_download(force_download=args.force_download, mode=args.mode)
    source = df.attrs.get("data_source", args.mode)
    print(f"Bars loaded: {len(df)} | {df.index.min()} -> {df.index.max()} | source={source}")

    print("Running backtest...")
    trades = run_backtest(df)
    trades_df = trades_to_dataframe(trades)

    if trades_df.empty:
        print("No trades generated. Check data or pattern rules.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trades_path = OUTPUT_DIR / f"trades_{args.mode}.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"Trades saved: {trades_path}")

    summary = compute_summary(trades, INITIAL_CAPITAL)

    r_multiples = trades_df["r_multiple"].to_numpy()
    mc_results = run_trade_monte_carlo(r_multiples)
    mc_stats = monte_carlo_summary(mc_results)

    print_summary(summary, mc_stats, trades_df, data_source=source)

    if not args.no_plots:
        plot_dir = OUTPUT_DIR / args.mode
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating plots -> {plot_dir}")
        generate_all_plots(df, trades_df, mc_results, INITIAL_CAPITAL, output_dir=plot_dir)
        print(f"Saved 9 chart files in {plot_dir}")

        if args.show:
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt

            for img_path in sorted(plot_dir.glob("*.png")):
                img = mpimg.imread(img_path)
                plt.figure(figsize=(12, 7))
                plt.imshow(img)
                plt.axis("off")
                plt.title(img_path.name)
            plt.show()


if __name__ == "__main__":
    main()
