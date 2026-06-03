import yfinance as yf

def get_data(ticker):

    df = yf.download(
        ticker,
        start="2010-01-01",
        auto_adjust=True
    )

    df.columns = df.columns.get_level_values(0)

    return df