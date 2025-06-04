import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Get ticker from user
ticker = input("Enter ticker of the stock ex: AAPL, MSFT, TSLA: ").upper()
print(f"Downloading data for {ticker}...")

# Download data
data = yf.download(ticker, start="2019-01-01", end="2024-12-31")
data = data[['Adj Close']].copy()
data.columns = ['Close']

# Calculate moving averages
data['SMA50'] = data['Close'].rolling(window=50).mean()
data['SMA200'] = data['Close'].rolling(window=200).mean()

# Generate trading signals
data['Signal'] = 0
data.loc[data['SMA50'] > data['SMA200'], 'Signal'] = 1

# Shift signals to avoid look-ahead bias
data['Position'] = data['Signal'].shift(1)

# Calculate returns
data['Returns'] = data['Close'].pct_change()

# Drop NaN values
data = data.dropna()

print(f"Data shape after cleaning: {data.shape}")
print(f"Date range: {data.index[0]} to {data.index[-1]}")

# DEBUG: Check signals
print(f"\nSignal distribution:")
print(data['Signal'].value_counts())
print(f"\nPosition distribution:")
print(data['Position'].value_counts())

# Calculate strategy returns
# When Position = 1: get market return
# When Position = 0: get 0% return
data['Strategy_Returns'] = data['Returns'] * data['Position']

# DEBUG: Print sample of returns
print(f"\nSample returns (first 10 non-zero strategy returns):")
sample_data = data[data['Strategy_Returns'] != 0].head(10)
print("Date | Market_Return | Position | Strategy_Return")
print("-" * 60)
for date, row in sample_data.iterrows():
    print(f"{date.date()} | {row['Returns']:8.4f} | {row['Position']:8.0f} | {row['Strategy_Returns']:8.4f}")

# Calculate cumulative returns
data['Cumulative_BH'] = (1 + data['Returns']).cumprod()
data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()

# DEBUG: Print last 10 cumulative values
print(f"\nLast 10 cumulative returns:")
print("Date | BH_Cumulative | Strategy_Cumulative")
print("-" * 50)
for date, row in data.tail(10).iterrows():
    print(f"{date.date()} | {row['Cumulative_BH']:12.3f} | {row['Cumulative_Strategy']:12.3f}")

# Plot results
plt.figure(figsize=(14, 8))
plt.plot(data.index, data['Cumulative_BH'], label='Buy & Hold', linewidth=2)
plt.plot(data.index, data['Cumulative_Strategy'], label='Moving Average Strategy', linewidth=2)
plt.title(f'{ticker} - Moving Average Strategy vs Buy & Hold')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns (Multiple)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Final results
final_bh = data['Cumulative_BH'].iloc[-1]
final_strategy = data['Cumulative_Strategy'].iloc[-1]

print(f"\n{'='*60}")
print(f"FINAL RESULTS FOR {ticker}")
print(f"{'='*60}")
print(f"Buy and Hold Final Return: {final_bh:.2f}x ({(final_bh-1)*100:.1f}%)")
print(f"Strategy Final Return: {final_strategy:.2f}x ({(final_strategy-1)*100:.1f}%)")

# Additional metrics
total_days = len(data)
days_in_market = data['Position'].sum()
print(f"\nStrategy was in market {int(days_in_market)}/{total_days} days ({days_in_market/total_days*100:.1f}%)")

# Count actual trades
position_changes = (data['Position'] != data['Position'].shift(1)).sum()
print(f"Number of position changes: {position_changes}")

# Show when strategy beats buy & hold
outperformance = data['Cumulative_Strategy'] > data['Cumulative_BH']
outperform_days = outperformance.sum()
print(f"Strategy outperformed on {outperform_days}/{total_days} days ({outperform_days/total_days*100:.1f}%)")

# Detailed breakdown by year
print(f"\nYearly Performance Breakdown:")
print("Year | Buy&Hold | Strategy | Difference")
print("-" * 45)
for year in range(2019, 2025):
    year_data = data[data.index.year == year]
    if len(year_data) > 0:
        start_bh = year_data['Cumulative_BH'].iloc[0] if len(year_data) > 0 else 1
        end_bh = year_data['Cumulative_BH'].iloc[-1] if len(year_data) > 0 else 1
        start_strat = year_data['Cumulative_Strategy'].iloc[0] if len(year_data) > 0 else 1
        end_strat = year_data['Cumulative_Strategy'].iloc[-1] if len(year_data) > 0 else 1
        
        if len(year_data) > 1:  # Need at least 2 data points
            yearly_bh = (end_bh / start_bh - 1) * 100
            yearly_strat = (end_strat / start_strat - 1) * 100
            diff = yearly_strat - yearly_bh
            print(f"{year} | {yearly_bh:8.1f}% | {yearly_strat:8.1f}% | {diff:+8.1f}%")