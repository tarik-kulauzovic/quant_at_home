import numpy as np
import matplotlib.pyplot as plt


def run_monte_carlo(df, simulations=10_000):
    strategy_returns = df["Strategy"].dropna().values

    all_equity_curves = []

    for _ in range(simulations):
        sampled_returns = np.random.choice(
            strategy_returns,
            size=len(strategy_returns),
            replace=True
        )

        equity_curve = np.cumprod(1 + sampled_returns)
        all_equity_curves.append(equity_curve)

    return np.array(all_equity_curves)


def plot_monte_carlo(mc_results):

    import numpy as np
    import matplotlib.pyplot as plt

    days = np.arange(mc_results.shape[1])

    p5 = np.percentile(mc_results, 5, axis=0)
    p50 = np.percentile(mc_results, 50, axis=0)
    p95 = np.percentile(mc_results, 95, axis=0)

    plt.figure(figsize=(13, 7))

    # Plot only 200 random paths so chart is cleaner
    for curve in mc_results[:200]:
        plt.plot(days, curve, alpha=0.04, linewidth=0.8)

    # Percentile range
    plt.fill_between(days, p5, p95, alpha=0.25, label="5% to 95% range")

    # Median path
    plt.plot(days, p50, linewidth=3, label="Median outcome")

    plt.axhline(1, linestyle="--", linewidth=1)

    plt.title("Monte Carlo Simulation — Strategy Equity Paths", fontsize=16)
    plt.xlabel("Trading days")
    plt.ylabel("Equity multiple")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def monte_carlo_stats(mc_results):
    final_values = mc_results[:, -1]

    print("Monte Carlo Results")
    print("-------------------")
    print("Worst case:", round(np.min(final_values), 2))
    print("5% percentile:", round(np.percentile(final_values, 5), 2))
    print("Median:", round(np.percentile(final_values, 50), 2))
    print("95% percentile:", round(np.percentile(final_values, 95), 2))
    print("Best case:", round(np.max(final_values), 2))


def plot_monte_carlo_histogram(mc_results):

    import matplotlib.pyplot as plt
    import numpy as np

    final_values = mc_results[:, -1]

    p5 = np.percentile(final_values, 5)
    p50 = np.percentile(final_values, 50)
    p95 = np.percentile(final_values, 95)

    plt.figure(figsize=(11, 6))

    plt.hist(final_values, bins=70, alpha=0.8)

    plt.axvline(p5, linestyle="--", linewidth=2, label=f"5%: {p5:.2f}")
    plt.axvline(p50, linestyle="-", linewidth=3, label=f"Median: {p50:.2f}")
    plt.axvline(p95, linestyle="--", linewidth=2, label=f"95%: {p95:.2f}")

    plt.title("Monte Carlo Final Equity Distribution", fontsize=16)
    plt.xlabel("Final equity multiple")
    plt.ylabel("Frequency")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def run_trade_monte_carlo(trades, simulations=10_000):

    import numpy as np

    trade_returns = np.array(
        [float(trade["Return"]) for trade in trades]
    )

    if len(trade_returns) == 0:
        raise ValueError("No trades found. Your strategy did not create any completed trades.")

    results = []

    for _ in range(simulations):

        sampled_trades = np.random.choice(
            trade_returns,
            size=len(trade_returns),
            replace=True
        )

        equity = np.cumprod(1 + sampled_trades)

        results.append(equity)

    return np.array(results)

def plot_trade_monte_carlo(mc_results):

    p5 = np.percentile(mc_results, 5, axis=0)
    p50 = np.percentile(mc_results, 50, axis=0)
    p95 = np.percentile(mc_results, 95, axis=0)

    x = np.arange(mc_results.shape[1])

    plt.figure(figsize=(13, 7))

    for curve in mc_results[:300]:
        plt.plot(x, curve, alpha=0.04, linewidth=0.8)

    plt.fill_between(x, p5, p95, alpha=0.25, label="5% to 95% range")
    plt.plot(x, p50, linewidth=3, label="Median")
    plt.axhline(1, linestyle="--", linewidth=1)

    plt.title("Trade-Based Monte Carlo Simulation")
    plt.xlabel("Trade Number")
    plt.ylabel("Equity Multiple")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def trade_monte_carlo_stats(mc_results):

    final = mc_results[:, -1]

    print("Trade Monte Carlo Results")
    print("-------------------------")
    print("Worst:", round(np.min(final), 2))
    print("5%:", round(np.percentile(final, 5), 2))
    print("Median:", round(np.percentile(final, 50), 2))
    print("95%:", round(np.percentile(final, 95), 2))
    print("Best:", round(np.max(final), 2))