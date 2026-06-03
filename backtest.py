import matplotlib.pyplot as plt


def plot_equity(df):
    equity = (1 + df["Strategy"]).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, equity)

    plt.title("Strategy Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")

    plt.grid()
    plt.show()


def extract_trades(df):
    trades = []
    in_position = False
    entry_price = None
    entry_date = None

    for i in range(len(df)):
        signal = df["Signal"].iloc[i]
        price = float(df["Close"].iloc[i])
        date = df.index[i]

        if not in_position and signal == 1:
            in_position = True
            entry_price = price
            entry_date = date

        elif in_position and signal == 0:
            exit_price = price
            exit_date = date

            trade_return = (exit_price / entry_price) - 1

            trades.append({
                "Entry Date": entry_date,
                "Exit Date": exit_date,
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "Return": trade_return
            })

            in_position = False

    return trades