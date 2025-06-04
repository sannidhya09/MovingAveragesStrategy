import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Moving Average Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.performance-section {
    background-color: #1e1e1e;
    padding: 2rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    color: white;
}

.performance-section h2 {
    color: white;
    margin-bottom: 1.5rem;
    text-align: center;
}

.performance-column {
    background-color: #2d2d30;
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin: 0.5rem;
    border: 1px solid #404040;
    color: white;
}

.performance-column h3 {
    color: white;
    margin-bottom: 1rem;
    text-align: center;
    border-bottom: 2px solid #404040;
    padding-bottom: 0.5rem;
}

.metric-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.8rem 0;
    border-bottom: 1px solid #404040;
    color: white;
}

.metric-item:last-child {
    border-bottom: none;
}

.metric-label {
    font-weight: 500;
    color: #cccccc;
}

.metric-value {
    font-weight: bold;
    font-size: 1.1em;
    color: white;
}

.metric-delta-positive {
    color: #00ff00;
    font-weight: bold;
}

.metric-delta-negative {
    color: #ff4444;
    font-weight: bold;
}

.metric-delta-neutral {
    color: #cccccc;
    font-weight: bold;
}

/* Override Streamlit's default metric styling when inside performance section */
.performance-section .stMetric {
    background-color: transparent !important;
    border: none !important;
    color: white !important;
}

.performance-section .stMetric > div {
    color: white !important;
}

.performance-section .stMetric label {
    color: #cccccc !important;
}

.performance-section .stMetric div[data-testid="metric-container"] {
    background-color: transparent !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

def load_data(ticker, start_date="2019-01-01", end_date="2024-12-31"):
    """Load and clean stock data"""
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if stock_data.empty:
            return None
        
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
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_indicators(df, sma_short=50, sma_long=200, ema_short=12, ema_long=26):
    """Calculate technical indicators"""
    df = df.copy()
    
    # Moving averages
    df['sma_short'] = df['price'].rolling(sma_short).mean()
    df['sma_long'] = df['price'].rolling(sma_long).mean()
    df['ema_short'] = df['price'].ewm(span=ema_short).mean()
    df['ema_long'] = df['price'].ewm(span=ema_long).mean()
    
    # Signals
    df['sma_signal'] = np.where(df['sma_short'] > df['sma_long'], 1, 0)
    df['ema_signal'] = np.where(df['ema_short'] > df['ema_long'], 1, 0)
    
    # Returns
    df['daily_return'] = df['price'].pct_change()
    df['sma_strategy_return'] = df['daily_return'] * df['sma_signal']
    df['ema_strategy_return'] = df['daily_return'] * df['ema_signal']
    
    # Cumulative performance
    df['buyhold_cumulative'] = (1 + df['daily_return']).cumprod()
    df['sma_strategy_cumulative'] = (1 + df['sma_strategy_return']).cumprod()
    df['ema_strategy_cumulative'] = (1 + df['ema_strategy_return']).cumprod()
    
    return df.dropna()

def calculate_metrics(df):
    """Calculate performance metrics"""
    metrics = {}
    
    # Final returns
    metrics['buyhold_return'] = (df['buyhold_cumulative'].iloc[-1] - 1) * 100
    metrics['sma_return'] = (df['sma_strategy_cumulative'].iloc[-1] - 1) * 100
    metrics['ema_return'] = (df['ema_strategy_cumulative'].iloc[-1] - 1) * 100
    
    # Volatility
    metrics['buyhold_vol'] = df['daily_return'].std() * np.sqrt(252) * 100
    metrics['sma_vol'] = df['sma_strategy_return'].std() * np.sqrt(252) * 100
    metrics['ema_vol'] = df['ema_strategy_return'].std() * np.sqrt(252) * 100
    
    # Sharpe ratios
    metrics['buyhold_sharpe'] = (df['daily_return'].mean() * 252) / (df['daily_return'].std() * np.sqrt(252)) if df['daily_return'].std() > 0 else 0
    metrics['sma_sharpe'] = (df['sma_strategy_return'].mean() * 252) / (df['sma_strategy_return'].std() * np.sqrt(252)) if df['sma_strategy_return'].std() > 0 else 0
    metrics['ema_sharpe'] = (df['ema_strategy_return'].mean() * 252) / (df['ema_strategy_return'].std() * np.sqrt(252)) if df['ema_strategy_return'].std() > 0 else 0
    
    # Drawdowns
    buyhold_peak = df['buyhold_cumulative'].expanding().max()
    buyhold_drawdown = (df['buyhold_cumulative'] / buyhold_peak - 1) * 100
    sma_peak = df['sma_strategy_cumulative'].expanding().max()
    sma_drawdown = (df['sma_strategy_cumulative'] / sma_peak - 1) * 100
    ema_peak = df['ema_strategy_cumulative'].expanding().max()
    ema_drawdown = (df['ema_strategy_cumulative'] / ema_peak - 1) * 100
    
    metrics['buyhold_max_dd'] = buyhold_drawdown.min()
    metrics['sma_max_dd'] = sma_drawdown.min()
    metrics['ema_max_dd'] = ema_drawdown.min()
    
    # Trade analysis
    sma_trades = df['sma_strategy_return'][df['sma_strategy_return'] != 0]
    ema_trades = df['ema_strategy_return'][df['ema_strategy_return'] != 0]
    
    metrics['sma_win_rate'] = len(sma_trades[sma_trades > 0]) / len(sma_trades) * 100 if len(sma_trades) > 0 else 0
    metrics['ema_win_rate'] = len(ema_trades[ema_trades > 0]) / len(ema_trades) * 100 if len(ema_trades) > 0 else 0
    metrics['sma_time_in_market'] = df['sma_signal'].mean() * 100
    metrics['ema_time_in_market'] = df['ema_signal'].mean() * 100
    
    return metrics, buyhold_drawdown, sma_drawdown, ema_drawdown

def create_price_chart(df, ticker, sma_short, sma_long):
    """Create price and moving averages chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['price'],
        mode='lines', name='Price',
        line=dict(color='black', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['sma_short'],
        mode='lines', name=f'SMA {sma_short}',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['sma_long'],
        mode='lines', name=f'SMA {sma_long}',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f'{ticker} - Price with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_performance_chart(df):
    """Create cumulative performance comparison chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['buyhold_cumulative'],
        mode='lines', name='Buy & Hold',
        line=dict(color='green', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['sma_strategy_cumulative'],
        mode='lines', name='SMA Strategy',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['ema_strategy_cumulative'],
        mode='lines', name='EMA Strategy',
        line=dict(color='orange', width=3)
    ))
    
    fig.update_layout(
        title='Cumulative Returns Comparison',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($1 initial)',
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_drawdown_chart(df, buyhold_drawdown, sma_drawdown, ema_drawdown):
    """Create drawdown chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=buyhold_drawdown,
        mode='lines', name='Buy & Hold',
        fill='tonexty', fillcolor='rgba(0,128,0,0.3)',
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=sma_drawdown,
        mode='lines', name='SMA Strategy',
        fill='tonexty', fillcolor='rgba(0,0,255,0.3)',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=ema_drawdown,
        mode='lines', name='EMA Strategy',
        fill='tonexty', fillcolor='rgba(255,165,0,0.3)',
        line=dict(color='orange')
    ))
    
    fig.update_layout(
        title='Portfolio Drawdowns',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_rolling_sharpe_chart(df):
    """Create rolling Sharpe ratio chart"""
    window = 252
    if len(df) > window:
        buyhold_rolling_sharpe = df['daily_return'].rolling(window).mean() / df['daily_return'].rolling(window).std() * np.sqrt(252)
        sma_rolling_sharpe = df['sma_strategy_return'].rolling(window).mean() / df['sma_strategy_return'].rolling(window).std() * np.sqrt(252)
        ema_rolling_sharpe = df['ema_strategy_return'].rolling(window).mean() / df['ema_strategy_return'].rolling(window).std() * np.sqrt(252)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index, y=buyhold_rolling_sharpe,
            mode='lines', name='Buy & Hold',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=sma_rolling_sharpe,
            mode='lines', name='SMA Strategy',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=ema_rolling_sharpe,
            mode='lines', name='EMA Strategy',
            line=dict(color='orange')
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title='Rolling Sharpe Ratio (1-Year Window)',
            xaxis_title='Date',
            yaxis_title='Sharpe Ratio',
            hovermode='x unified',
            height=400
        )
        
        return fig
    return None

def display_custom_metrics(col, title, return_val, volatility, sharpe, max_dd, delta_return=None):
    """Display custom styled metrics"""
    delta_class = ""
    delta_text = ""
    if delta_return is not None:
        if delta_return > 0:
            delta_class = "metric-delta-positive"
            delta_text = f"(+{delta_return:.1f}%)"
        elif delta_return < 0:
            delta_class = "metric-delta-negative"
            delta_text = f"({delta_return:.1f}%)"
        else:
            delta_class = "metric-delta-neutral"
            delta_text = "(0.0%)"
    
    with col:
        st.markdown(f"""
        <div class="performance-column">
            <h3>{title}</h3>
            <div class="metric-item">
                <span class="metric-label">Return</span>
                <span class="metric-value">{return_val:.1f}% <span class="{delta_class}">{delta_text}</span></span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Volatility</span>
                <span class="metric-value">{volatility:.1f}%</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Sharpe Ratio</span>
                <span class="metric-value">{sharpe:.2f}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Max Drawdown</span>
                <span class="metric-value">{max_dd:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main Streamlit App
def main():
    st.title("📈 Moving Average Strategy Backtester")
    st.markdown("Compare Buy & Hold vs SMA vs EMA trading strategies")
    
    # Sidebar for inputs
    st.sidebar.header("📊 Strategy Parameters")
    
    # Stock selection
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL", help="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)")
    ticker = ticker.upper()
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start Date", value=pd.to_datetime("2019-01-01"))
    end_date = col2.date_input("End Date", value=pd.to_datetime("2024-12-31"))
    
    st.sidebar.subheader("Moving Average Parameters")
    sma_short = st.sidebar.slider("Short SMA Period", min_value=5, max_value=100, value=50)
    sma_long = st.sidebar.slider("Long SMA Period", min_value=50, max_value=300, value=200)
    ema_short = st.sidebar.slider("Short EMA Period", min_value=5, max_value=50, value=12)
    ema_long = st.sidebar.slider("Long EMA Period", min_value=15, max_value=100, value=26)
    
    # Analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        if ticker:
            with st.spinner(f"Loading data for {ticker}..."):
                # Load data
                df = load_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                
                if df is not None:
                    # Calculate indicators
                    df = calculate_indicators(df, sma_short, sma_long, ema_short, ema_long)
                    
                    if len(df) > 0:
                        # Calculate metrics
                        metrics, buyhold_dd, sma_dd, ema_dd = calculate_metrics(df)
                        
                        # Display results
                        st.success(f"✅ Analysis completed for {ticker} ({len(df)} trading days)")
                        
                        # Performance metrics with dark background
                        st.markdown("""
                        <div class="performance-section">
                            <h2>📊 Performance Summary</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        # Calculate deltas
                        delta_sma = metrics['sma_return'] - metrics['buyhold_return']
                        delta_ema = metrics['ema_return'] - metrics['buyhold_return']
                        
                        # Display custom metrics
                        display_custom_metrics(
                            col1, "📊 Buy & Hold",
                            metrics['buyhold_return'],
                            metrics['buyhold_vol'],
                            metrics['buyhold_sharpe'],
                            metrics['buyhold_max_dd']
                        )
                        
                        display_custom_metrics(
                            col2, "📈 SMA Strategy",
                            metrics['sma_return'],
                            metrics['sma_vol'],
                            metrics['sma_sharpe'],
                            metrics['sma_max_dd'],
                            delta_sma
                        )
                        
                        display_custom_metrics(
                            col3, "📊 EMA Strategy",
                            metrics['ema_return'],
                            metrics['ema_vol'],
                            metrics['ema_sharpe'],
                            metrics['ema_max_dd'],
                            delta_ema
                        )
                        
                        # Add some spacing after the performance section
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Charts
                        st.header("📈 Interactive Charts")
                        
                        # Price chart
                        st.subheader("Price and Moving Averages")
                        price_chart = create_price_chart(df, ticker, sma_short, sma_long)
                        st.plotly_chart(price_chart, use_container_width=True)
                        
                        # Performance chart
                        st.subheader("Cumulative Performance Comparison")
                        perf_chart = create_performance_chart(df)
                        st.plotly_chart(perf_chart, use_container_width=True)
                        
                        # Two column layout for additional charts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Portfolio Drawdowns")
                            dd_chart = create_drawdown_chart(df, buyhold_dd, sma_dd, ema_dd)
                            st.plotly_chart(dd_chart, use_container_width=True)
                        
                        with col2:
                            st.subheader("Rolling Sharpe Ratio")
                            sharpe_chart = create_rolling_sharpe_chart(df)
                            if sharpe_chart:
                                st.plotly_chart(sharpe_chart, use_container_width=True)
                            else:
                                st.info("Not enough data for rolling Sharpe ratio calculation")
                        
                        # Strategy details
                        st.header("🎯 Strategy Details")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("SMA Strategy")
                            st.write(f"**Time in Market:** {metrics['sma_time_in_market']:.1f}%")
                            st.write(f"**Win Rate:** {metrics['sma_win_rate']:.1f}%")
                            sma_signal_changes = (df['sma_signal'] != df['sma_signal'].shift(1)).sum()
                            st.write(f"**Signal Changes:** {sma_signal_changes}")
                        
                        with col2:
                            st.subheader("EMA Strategy")
                            st.write(f"**Time in Market:** {metrics['ema_time_in_market']:.1f}%")
                            st.write(f"**Win Rate:** {metrics['ema_win_rate']:.1f}%")
                            ema_signal_changes = (df['ema_signal'] != df['ema_signal'].shift(1)).sum()
                            st.write(f"**Signal Changes:** {ema_signal_changes}")
                        
                        # Recent signals
                        st.header("🔍 Recent Signals")
                        recent_data = df[['price', 'sma_short', 'sma_long', 'ema_short', 'ema_long', 'sma_signal', 'ema_signal']].tail(10)
                        recent_data.columns = ['Price', f'SMA {sma_short}', f'SMA {sma_long}', f'EMA {ema_short}', f'EMA {ema_long}', 'SMA Signal', 'EMA Signal']
                        
                        # Format the dataframe
                        recent_formatted = recent_data.copy()
                        for col in ['Price', f'SMA {sma_short}', f'SMA {sma_long}', f'EMA {ema_short}', f'EMA {ema_long}']:
                            recent_formatted[col] = recent_formatted[col].apply(lambda x: f"${x:.2f}")
                        recent_formatted['SMA Signal'] = recent_formatted['SMA Signal'].apply(lambda x: "BUY" if x == 1 else "SELL")
                        recent_formatted['EMA Signal'] = recent_formatted['EMA Signal'].apply(lambda x: "BUY" if x == 1 else "SELL")
                        
                        st.dataframe(recent_formatted, use_container_width=True)
                        
                        # Recommendation
                        st.header("🏆 Recommendation")
                        strategies = {
                            'Buy & Hold': metrics['buyhold_return'],
                            'SMA Strategy': metrics['sma_return'],
                            'EMA Strategy': metrics['ema_return']
                        }
                        best_strategy = max(strategies, key=strategies.get)
                        
                        if best_strategy == 'Buy & Hold':
                            st.info(f"🎯 **{best_strategy}** performed best with {strategies[best_strategy]:.1f}% return")
                        else:
                            st.success(f"🎯 **{best_strategy}** performed best with {strategies[best_strategy]:.1f}% return")
                    
                    else:
                        st.error("Not enough data after applying indicators. Try a longer date range.")
                else:
                    st.error(f"Could not load data for {ticker}. Please check the ticker symbol.")
        else:
            st.warning("Please enter a stock ticker symbol.")

if __name__ == "__main__":
    main()