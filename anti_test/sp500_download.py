import yfinance as yf
from datetime import datetime

# Define the ticker for the S&P 500 index (using the ^GSPC ticker)
sp500 = yf.Ticker("^GSPC")

# Define the start and end dates
start_date = "2022-01-01"
# End date is today
end_date = datetime.now().strftime("%Y-%m-%d")

# Download historical daily price data
historical = sp500.history(start=start_date, end=end_date)

# Optionally, reset index to have Date as a column
historical = historical.reset_index()

# Save to CSV file in the same directory
output_path = "sp500_data.csv"
historical.to_csv(output_path, index=False)

print(f"Downloaded {len(historical)} rows of S&P 500 data from {start_date} to {end_date} and saved to {output_path}")

# Plot the closing price
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(historical['Date'], historical['Close'], label='S&P 500 Close')
plt.title('S&P 500 Daily Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('sp500_plot.png')
plt.show()
