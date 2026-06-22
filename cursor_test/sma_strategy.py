import pandas as pd


def generate_signals(df):
    df = df.copy()

    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    df["Signal"] = 0
    df.loc[df["SMA50"] > df["SMA200"], "Signal"] = 1

    df["Returns"] = df["Close"].pct_change()
    df["Strategy"] = df["Signal"].shift(1) * df["Returns"]

    return df


def extract_trades(df):
    trades = []
    in_position = False
    entry_price = None
    entry_date = None

    for i in range(len(df)):
        signal = df["Signal"].iloc[i]
        price = df["Close"].iloc[i]
        date = df.index[i]

        if not in_position and signal == 1:
            in_position = True
            entry_price = price
            entry_date = date

        elif in_position and signal == 0:
            trades.append({
                "Entry Date": entry_date,
                "Exit Date": date,
                "Entry Price": entry_price,
                "Exit Price": price,
                "Return": (price / entry_price) - 1,
            })
            in_position = False

    return trades


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = pd.read_csv("nvda_data.csv", parse_dates=["Date"])
    df = df.set_index("Date")

    df = generate_signals(df)
    trades = extract_trades(df)

    buy_hold = (1 + df["Returns"].fillna(0)).cumprod()
    strategy = (1 + df["Strategy"].fillna(0)).cumprod()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(df.index, df["Close"], label="Close", linewidth=1)
    ax1.plot(df.index, df["SMA50"], label="SMA50", linewidth=1)
    ax1.plot(df.index, df["SMA200"], label="SMA200", linewidth=1)
    ax1.set_title("NVDA with SMA50 / SMA200")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df.index, buy_hold, label="Buy & Hold")
    ax2.plot(df.index, strategy, label="SMA50/200 Strategy")
    ax2.set_title("Equity Curve")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Growth of $1")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("sma_strategy_plot.png")
    plt.show()

    print(f"Trades: {len(trades)}")
    print(f"Strategy return: {(strategy.iloc[-1] - 1) * 100:.2f}%")
    print(f"Buy & hold return: {(buy_hold.iloc[-1] - 1) * 100:.2f}%")
