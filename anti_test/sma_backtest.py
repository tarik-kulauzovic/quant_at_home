import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None

def fetch_sp500_data(start_date: str = "2022-01-01") -> pd.DataFrame:
    """Download daily S&P 500 (^GSPC) price data from Yahoo Finance.

    Parameters
    ----------
    start_date: str
        Start date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame with Date as index and a single column 'Close'.
    """
    ticker = "^GSPC"
    df = yf.download(ticker, start=start_date, progress=False)
    df = df[['Adj Close']].rename(columns={'Adj Close': 'Close'})
    df.index = pd.to_datetime(df.index)
    return df

def compute_sma(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculate simple moving average for the given window."""
    return df['Close'].rolling(window=window).mean()

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate SMA‑50 / SMA‑200 crossover signals.

    Long (1) when SMA‑50 > SMA‑200, otherwise flat (0).
    """
    df = df.copy()
    df['SMA50'] = compute_sma(df, 50)
    df['SMA200'] = compute_sma(df, 200)
    df['Signal'] = 0
    df.loc[df['SMA50'] > df['SMA200'], 'Signal'] = 1
    df['Position'] = df['Signal'].diff()  # +1 entry, -1 exit
    return df

def backtest_strategy(df: pd.DataFrame, initial_capital: float = 100_000) -> pd.DataFrame:
    """Backtest the SMA crossover strategy.

    The portfolio is fully invested on a long signal and fully in cash otherwise.
    """
    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    # Apply yesterday's signal to today's return
    df['Strategy'] = df['Signal'].shift(1) * df['Returns']
    df['Strategy'].fillna(0, inplace=True)
    df['Equity'] = (1 + df['Strategy']).cumprod() * initial_capital
    return df

def monte_carlo_simulation(df: pd.DataFrame, n_sim: int = 1000, days: int = None, seed: int = 42) -> pd.DataFrame:
    """Monte‑Carlo simulation of the strategy using geometric Brownian motion.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data with a 'Returns' column.
    n_sim : int
        Number of simulated paths.
    days : int, optional
        Length of each simulated path. Defaults to length of historical data.
    seed : int
        Random seed for reproducibility.
    """
    np.random.seed(seed)
    if days is None:
        days = len(df)
    mu = df['Returns'].mean()
    sigma = df['Returns'].std()

    final_equities = []
    for _ in range(n_sim):
        # Simulate daily returns
        simulated_returns = np.random.normal(loc=mu, scale=sigma, size=days)
        # Build a synthetic price series starting from last known close
        price_path = [df['Close'].iloc[-1]]
        for r in simulated_returns:
            price_path.append(price_path[-1] * (1 + r))
        sim_df = pd.DataFrame({'Close': price_path[1:]})
        sim_df = generate_signals(sim_df)
        sim_df['Returns'] = simulated_returns
        sim_df['Strategy'] = sim_df['Signal'].shift(1) * sim_df['Returns']
        sim_df['Strategy'].fillna(0, inplace=True)
        final_equity = (1 + sim_df['Strategy']).prod() * 100_000
        final_equities.append(final_equity)
    return pd.DataFrame({'FinalEquity': final_equities})

def plot_results(df: pd.DataFrame, mc_df: pd.DataFrame):
    """Plot price with SMAs, equity curve, and Monte‑Carlo distribution."""
    plt.style.use('ggplot')
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # Price + SMAs
    axs[0].plot(df.index, df['Close'], label='Close', color='steelblue')
    axs[0].plot(df.index, df['SMA50'], label='SMA 50', color='orange')
    axs[0].plot(df.index, df['SMA200'], label='SMA 200', color='green')
    axs[0].set_title('S&P 500 with SMA 50 / SMA 200')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price')
    axs[0].legend()

    # Equity curve
    axs[1].plot(df.index, df['Equity'], label='Strategy Equity', color='purple')
    axs[1].set_title('Equity Curve')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Portfolio Value')
    axs[1].legend()

    # Monte‑Carlo histogram
    axs[2].hist(mc_df['FinalEquity'], bins=30, color='teal', edgecolor='black')
    axs[2].set_title('Monte‑Carlo Simulation – Final Equity Distribution')
    axs[2].set_xlabel('Final Portfolio Value')
    axs[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def main():
    start_date = "2022-01-01"
    print(f"Downloading S&P 500 data from {start_date} …")
    df = fetch_sp500_data(start_date)
    df = generate_signals(df)
    df = backtest_strategy(df)
    print("Running Monte‑Carlo simulation (1,000 paths)…")
    mc_results = monte_carlo_simulation(df, n_sim=1000)
    plot_results(df, mc_results)

if __name__ == "__main__":
    main()
