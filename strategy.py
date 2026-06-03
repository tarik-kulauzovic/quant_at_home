def generate_signals(df):

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    df["Signal"] = 0
    df.loc[df["EMA20"] > df["EMA50"], "Signal"] = 1

    df["Returns"] = df["Close"].pct_change()

    # This is the important missing line
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

        # Enter trade
        if not in_position and signal == 1:
            in_position = True
            entry_price = price
            entry_date = date

        # Exit trade
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