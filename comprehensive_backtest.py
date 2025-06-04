import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def comprehensive_backtest():
    # Input
    ticker = input("Enter stock ticker for backtesting (e.g., AAPL, MSFT): ").upper()
    
    # Download data
    print(f"Downloading {ticker} data for comprehensive backtest...")
    stock_data = yf.download(ticker, start="2019-01-01", end="2024-12-31", progress=False)
    
    if stock_data.empty:
        print("No data found!")
        return
    
    # Create clean dataframe
    df = pd.DataFrame()
    
    # Handle different column formats
    if ('Close', ticker) in stock_data.columns:
        df['price'] = stock_data[('Close', ticker)]
        df['volume'] = stock_data[('Volume', ticker)]
        df['high'] = stock_data[('High', ticker)]
        df['low'] = stock_data[('Low', ticker)]
    else:
        df['price'] = stock_data['Close']
        df['volume'] = stock_data['Volume']
        df['high'] = stock_data['High']
        df['low'] = stock_data['Low']
    
    # Calculate moving averages
    df['sma_50'] = df['price'].rolling(50).mean()
    df['sma_200'] = df['price'].rolling(200).mean()
    df['ema_12'] = df['price'].ewm(span=12).mean()
    df['ema_26'] = df['price'].ewm(span=26).mean()
    
    # Remove NaN rows
    df = df.dropna().copy()
    
    print(f"Backtesting period: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")
    
    # Strategy signals
    df['ma_signal'] = np.where(df['sma_50'] > df['sma_200'], 1, 0)
    df['ema_signal'] = np.where(df['ema_12'] > df['ema_26'], 1, 0)
    
    # Calculate returns
    df['daily_return'] = df['price'].pct_change()
    df['ma_strategy_return'] = df['daily_return'] * df['ma_signal']
    df['ema_strategy_return'] = df['daily_return'] * df['ema_signal']
    
    # Remove first row with NaN
    df = df.dropna().copy()
    
    # Calculate cumulative performance
    df['buyhold_cumulative'] = (1 + df['daily_return']).cumprod()
    df['ma_strategy_cumulative'] = (1 + df['ma_strategy_return']).cumprod()
    df['ema_strategy_cumulative'] = (1 + df['ema_strategy_return']).cumprod()
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker} - Comprehensive Backtesting Analysis', fontsize=16, fontweight='bold')
    
    # 1. Price and Moving Averages
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['price'], label='Price', alpha=0.7, color='black')
    ax1.plot(df.index, df['sma_50'], label='SMA 50', color='blue')
    ax1.plot(df.index, df['sma_200'], label='SMA 200', color='red')
    ax1.set_title('Price with Moving Averages')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative Returns Comparison
    ax2 = axes[0, 1]
    ax2.plot(df.index, df['buyhold_cumulative'], label='Buy & Hold', linewidth=2, color='green')
    ax2.plot(df.index, df['ma_strategy_cumulative'], label='SMA Strategy', linewidth=2, color='blue')
    ax2.plot(df.index, df['ema_strategy_cumulative'], label='EMA Strategy', linewidth=2, color='orange')
    ax2.set_title('Cumulative Returns Comparison')
    ax2.set_ylabel('Portfolio Value ($1 initial)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Rolling Sharpe Ratio (252-day window)
    ax3 = axes[1, 0]
    
    # Calculate rolling Sharpe ratios
    window = 252
    if len(df) > window:
        buyhold_rolling_sharpe = df['daily_return'].rolling(window).mean() / df['daily_return'].rolling(window).std() * np.sqrt(252)
        ma_rolling_sharpe = df['ma_strategy_return'].rolling(window).mean() / df['ma_strategy_return'].rolling(window).std() * np.sqrt(252)
        
        ax3.plot(df.index, buyhold_rolling_sharpe, label='Buy & Hold', color='green', alpha=0.8)
        ax3.plot(df.index, ma_rolling_sharpe, label='SMA Strategy', color='blue', alpha=0.8)
        ax3.set_title('Rolling Sharpe Ratio (1-Year Window)')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. Drawdown Analysis
    ax4 = axes[1, 1]
    
    # Calculate drawdowns
    buyhold_peak = df['buyhold_cumulative'].expanding().max()
    buyhold_drawdown = (df['buyhold_cumulative'] / buyhold_peak - 1) * 100
    
    ma_peak = df['ma_strategy_cumulative'].expanding().max()
    ma_drawdown = (df['ma_strategy_cumulative'] / ma_peak - 1) * 100
    
    ax4.fill_between(df.index, buyhold_drawdown, 0, alpha=0.3, color='green', label='Buy & Hold')
    ax4.fill_between(df.index, ma_drawdown, 0, alpha=0.3, color='blue', label='SMA Strategy')
    ax4.set_title('Portfolio Drawdowns')
    ax4.set_ylabel('Drawdown (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance Statistics
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE BACKTEST RESULTS FOR {ticker}")
    print(f"{'='*80}")
    
    # Final returns
    final_buyhold = df['buyhold_cumulative'].iloc[-1]
    final_ma = df['ma_strategy_cumulative'].iloc[-1]
    final_ema = df['ema_strategy_cumulative'].iloc[-1]
    
    print(f"\n📊 TOTAL RETURNS:")
    print(f"Buy & Hold:        {final_buyhold:.3f}x ({(final_buyhold-1)*100:+.1f}%)")
    print(f"SMA Strategy:      {final_ma:.3f}x ({(final_ma-1)*100:+.1f}%)")
    print(f"EMA Strategy:      {final_ema:.3f}x ({(final_ema-1)*100:+.1f}%)")
    
    # Risk metrics
    buyhold_vol = df['daily_return'].std() * np.sqrt(252)
    ma_vol = df['ma_strategy_return'].std() * np.sqrt(252)
    ema_vol = df['ema_strategy_return'].std() * np.sqrt(252)
    
    print(f"\n📈 ANNUALIZED VOLATILITY:")
    print(f"Buy & Hold:        {buyhold_vol:.1%}")
    print(f"SMA Strategy:      {ma_vol:.1%}")
    print(f"EMA Strategy:      {ema_vol:.1%}")
    
    # Sharpe ratios
    buyhold_sharpe = (df['daily_return'].mean() * 252) / buyhold_vol if buyhold_vol > 0 else 0
    ma_sharpe = (df['ma_strategy_return'].mean() * 252) / ma_vol if ma_vol > 0 else 0
    ema_sharpe = (df['ema_strategy_return'].mean() * 252) / ema_vol if ema_vol > 0 else 0
    
    print(f"\n⚡ SHARPE RATIOS:")
    print(f"Buy & Hold:        {buyhold_sharpe:.2f}")
    print(f"SMA Strategy:      {ma_sharpe:.2f}")
    print(f"EMA Strategy:      {ema_sharpe:.2f}")
    
    # Maximum drawdowns
    max_buyhold_dd = buyhold_drawdown.min()
    max_ma_dd = ma_drawdown.min()
    
    print(f"\n📉 MAXIMUM DRAWDOWNS:")
    print(f"Buy & Hold:        {max_buyhold_dd:.1f}%")
    print(f"SMA Strategy:      {max_ma_dd:.1f}%")
    
    # Win rates and trade analysis
    ma_trades = df['ma_strategy_return'][df['ma_strategy_return'] != 0]
    wins = len(ma_trades[ma_trades > 0])
    total_trades = len(ma_trades)
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    
    print(f"\n🎯 TRADE ANALYSIS (SMA Strategy):")
    print(f"Total trading days: {total_trades}")
    print(f"Winning days:       {wins} ({win_rate:.1f}%)")
    print(f"Time in market:     {df['ma_signal'].mean()*100:.1f}%")
    
    signal_changes = (df['ma_signal'] != df['ma_signal'].shift(1)).sum()
    print(f"Signal changes:     {signal_changes}")
    print(f"Avg trade duration: {total_trades/(signal_changes/2):.0f} days" if signal_changes > 0 else "N/A")
    
    # Year-by-year performance
    print(f"\n📅 YEAR-BY-YEAR PERFORMANCE:")
    print("Year | Buy&Hold | SMA Strategy | EMA Strategy")
    print("-" * 50)
    
    for year in range(2019, 2025):
        year_data = df[df.index.year == year]
        if len(year_data) > 1:
            # Calculate yearly returns
            start_bh = year_data['buyhold_cumulative'].iloc[0]
            end_bh = year_data['buyhold_cumulative'].iloc[-1]
            yearly_bh = (end_bh / start_bh - 1) * 100
            
            start_ma = year_data['ma_strategy_cumulative'].iloc[0]
            end_ma = year_data['ma_strategy_cumulative'].iloc[-1]
            yearly_ma = (end_ma / start_ma - 1) * 100
            
            start_ema = year_data['ema_strategy_cumulative'].iloc[0]
            end_ema = year_data['ema_strategy_cumulative'].iloc[-1]
            yearly_ema = (end_ema / start_ema - 1) * 100
            
            print(f"{year} | {yearly_bh:+7.1f}% | {yearly_ma:+10.1f}% | {yearly_ema:+10.1f}%")
    
    # Best strategy recommendation
    print(f"\n🏆 RECOMMENDATION:")
    strategies = {
        'Buy & Hold': final_buyhold,
        'SMA Strategy': final_ma,
        'EMA Strategy': final_ema
    }
    best_strategy = max(strategies, key=strategies.get)
    print(f"Best performing strategy: {best_strategy} with {strategies[best_strategy]:.3f}x return")

if __name__ == "__main__":
    comprehensive_backtest()