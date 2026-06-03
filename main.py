from data import get_data
from strategy import generate_signals
from backtest import plot_equity, extract_trades
from monte_carlo import (
    run_trade_monte_carlo,
    plot_trade_monte_carlo,
    trade_monte_carlo_stats
)

df = get_data("SPY")

df = generate_signals(df)
print(df.columns)
plot_equity(df)

trades = extract_trades(df)

print("Number of trades:", len(trades))
print("First 5 trades:")
print(trades[:5])


mc_results = run_trade_monte_carlo(
    trades,
    simulations=10_000
)

plot_trade_monte_carlo(mc_results)
trade_monte_carlo_stats(mc_results)