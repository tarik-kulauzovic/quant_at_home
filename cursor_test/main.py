import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt

ticker = yf.Ticker("NVDA")

start_date = "2020-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

historical = ticker.history(start=start_date, end=end_date)
historical = historical.reset_index()

output_path = "nvda_data.csv"
historical.to_csv(output_path, index=False)

print(
    f"Downloaded {len(historical)} rows of NVDA data "
    f"from {start_date} to {end_date} and saved to {output_path}"
)

plt.figure(figsize=(12, 6))
plt.plot(historical["Date"], historical["Close"], label="NVDA Close")
plt.title("NVDA Daily Close Price (2020 onwards)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("nvda_plot.png")
plt.show()
