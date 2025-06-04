import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Input
    ticker = input("Enter stock ticker (e.g., AAPL, MSFT): ").upper()
    
    # Download data
    print(f"Downloading {ticker} data...")
    stock_data = yf.download(ticker, start="2019-01-01", end="2024-12-31", progress=False)
    
    if stock_data.empty:
        print("No data found!")
        return
    
    # Debug: Check what columns we have
    print(f"Available columns: {list(stock_data.columns)}")
    
    # Create clean dataframe - handle both single and multi-level columns
    df = pd.DataFrame()
    
    # Get the adjusted close price (handle different column formats)
    if ('Adj Close', ticker) in stock_data.columns:
        df['price'] = stock_data[('Adj Close', ticker)]
    elif 'Adj Close' in stock_data.columns:
        df['price'] = stock_data['Adj Close']
    else:
        # Fallback to regular Close
        if ('Close', ticker) in stock_data.columns:
            df['price'] = stock_data[('Close', ticker)]
        else:
            df['price'] = stock_data['Close']
    
    df['sma_50'] = df['price'].rolling(50).mean()
    df['sma_200'] = df['price'].rolling(200).mean()
    
    # Remove NaN rows
    df = df.dropna().copy()
    
    print(f"Analyzing {len(df)} trading days from {df.index[0].date()} to {df.index[-1].date()}")
    
    # Create signals: 1 = buy/hold, 0 = sell/cash
    df['signal'] = np.where(df['sma_50'] > df['sma_200'], 1, 0)
    
    # Check if we have any signal changes
    signal_changes = (df['signal'] != df['signal'].shift(1)).sum()
    print(f"Number of signal changes: {signal_changes}")
    
    if signal_changes == 0:
        print("Warning: No signal changes detected!")
        print("This means strategy will be either always in or always out of market")
    
    # Calculate daily returns
    df['daily_return'] = df['price'].pct_change()
    df = df.dropna().copy()
    
    # Strategy approach - multiply returns by signal
    # When signal = 1: get market return
    # When signal = 0: get 0% return
    df['strategy_return'] = df['daily_return'] * df['signal']
    
    # Calculate cumulative wealth (starting with $1)
    df['buyhold_wealth'] = (1 + df['daily_return']).cumprod()
    df['strategy_wealth'] = (1 + df['strategy_return']).cumprod()
    
    # Print debugging info
    print(f"\nDEBUG INFO:")
    print(f"Signal distribution: {df['signal'].value_counts().to_dict()}")
    print(f"Days in market: {df['signal'].sum()}/{len(df)} ({100*df['signal'].mean():.1f}%)")
    
    # Show some example calculations
    print(f"\nSample calculations (first 10 rows):")
    print("Date        | Signal | Daily Ret | Strategy Ret | BH Wealth | Strat Wealth")
    print("-" * 75)
    for i, (date, row) in enumerate(df.head(10).iterrows()):
        print(f"{date.date()} |   {row['signal']:1.0f}    | {row['daily_return']:8.4f} | {row['strategy_return']:11.4f} | {row['buyhold_wealth']:8.4f} | {row['strategy_wealth']:11.4f}")
    
    # Show final values
    final_buyhold = df['buyhold_wealth'].iloc[-1]
    final_strategy = df['strategy_wealth'].iloc[-1]
    
    print(f"\nLast 5 cumulative wealth values:")
    print("Date        | Buy&Hold | Strategy")
    print("-" * 35)
    for date, row in df.tail().iterrows():
        print(f"{date.date()} | {row['buyhold_wealth']:8.4f} | {row['strategy_wealth']:8.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(df.index, df['buyhold_wealth'], label='Buy & Hold', linewidth=2, color='blue')
    plt.plot(df.index, df['strategy_wealth'], label='Moving Average Strategy', linewidth=2, color='red')
    plt.title(f'{ticker} - Strategy Performance Comparison\n(SMA 50 vs SMA 200)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($1 initial investment)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Results
    print(f"\n" + "="*60)
    print(f"FINAL RESULTS FOR {ticker}")
    print(f"="*60)
    print(f"Buy & Hold Final Value:  ${final_buyhold:.4f} ({100*(final_buyhold-1):+.1f}% total return)")
    print(f"MA Strategy Final Value: ${final_strategy:.4f} ({100*(final_strategy-1):+.1f}% total return)")
    
    outperformance = (final_strategy / final_buyhold - 1) * 100
    if final_strategy > final_buyhold:
        print(f"✅ Strategy BEAT buy & hold by {outperformance:+.1f}%")
    else:
        print(f"❌ Strategy LOST to buy & hold by {outperformance:+.1f}%")
    
    # Additional analysis
    print(f"\nSTRATEGY DETAILS:")
    trades = signal_changes // 2
    print(f"Number of complete buy/sell cycles: {trades}")
    print(f"Percentage of time in market: {100*df['signal'].mean():.1f}%")
    
    # Show recent signals
    print(f"\nRecent signals (last 10 days):")
    recent = df[['price', 'sma_50', 'sma_200', 'signal']].tail(10)
    print("Date        | Price   | SMA50   | SMA200  | Signal")
    print("-" * 55)
    for date, row in recent.iterrows():
        signal_text = "BUY " if row['signal'] == 1 else "CASH"
        print(f"{date.date()} | ${row['price']:6.2f} | ${row['sma_50']:6.2f} | ${row['sma_200']:6.2f} | {signal_text}")

if __name__ == "__main__":
    main()