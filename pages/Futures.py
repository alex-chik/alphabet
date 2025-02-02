import streamlit as st
import vectorbt as vbt
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
from typing import Optional

st.set_page_config(page_title="Futures", layout="wide")

# Constants
DEFAULT_INITIAL_EQUITY = 100000
DEFAULT_POSITION_SIZE = "100"
DEFAULT_FEES = 0.00

STRATEGIES = {
    "SMA (9 & 21)": {"type": "SMA"},
    "EMA (5 & 13)": {"type": "EMA"},
    "RSI (Period 7, 25/75)": {"type": "RSI"},
}

def get_strategy_signals(data: pd.Series, strategy: str):
    """Generate entry and exit signals for futures scalping based on a 1-minute price series."""
    if strategy == "SMA":
        # Use a fast SMA (9 periods) and a slow SMA (21 periods)
        fast = vbt.MA.run(data, window=9)
        slow = vbt.MA.run(data, window=21)
        entries = fast.ma_crossed_above(slow)
        exits = fast.ma_crossed_below(slow)
    elif strategy == "EMA":
        # Use a fast EMA (5 periods) and a slow EMA (13 periods)
        fast = vbt.MA.run(data, window=5, ewm=True)
        slow = vbt.MA.run(data, window=13, ewm=True)
        entries = fast.ma_crossed_above(slow)
        exits = fast.ma_crossed_below(slow)
    elif strategy == "RSI":
        # Use an RSI with period 7; entry when RSI < 25 and exit when RSI > 75
        rsi = vbt.RSI.run(data, window=7)
        entries = rsi.rsi_below(25)
        exits = rsi.rsi_above(75)
    else:
        raise ValueError("Unknown strategy type provided.")
    
    return entries, exits

@st.cache_data
def fetch_historical_data(ticker: str) -> Optional[pd.Series]:
    """Fetch historical 1-minute interval price data for the past 8 days."""
    try:
        data = vbt.YFData.download(ticker, interval="1m", period="8d").get('Close')
        if data is None or data.empty:
            st.error(f"No data available for {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def plot_price_chart(ticker_data: pd.Series, ticker: str) -> go.Figure:
    """Create a chart showing the price for the ticker."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ticker_data.index, 
        y=ticker_data.values, 
        mode='lines', 
        name=ticker
    ))
    fig.update_layout(
        title=f"{ticker} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    return fig

def run_backtest(data: pd.Series, entries: pd.Series, exits: pd.Series, 
                 size: float, fees: float, initial_equity: float) -> vbt.Portfolio:
    """Run a backtest on the provided data with given parameters."""
    return vbt.Portfolio.from_signals(
        data, entries, exits,
        direction='longonly',
        size=size,
        size_type='percent',
        fees=fees / 100,
        init_cash=initial_equity,
        freq='1m',
        min_size=1,
        size_granularity=1
    )

def plot_equity_curve(equity: pd.Series, strategy: str) -> go.Figure:
    """Create a chart showing the equity curve for the ticker."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, 
        y=equity.values, 
        mode='lines', 
        name=strategy
    ))
    fig.update_layout(
        title=f"Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity"
    )
    return fig

def main():
    """Main application function."""
    # Sidebar controls
    with st.sidebar:
        st.header("Historical")
        ticker_options = ["ES=F", "NQ=F", "YM=F", "CL=F", "GC=F", "SI=F", "MNQ=F"]# You can add more ticker symbols as needed.
        ticker = st.selectbox("Ticker", options=ticker_options)
        historical_clicked = st.button("Last 8 Days")

        st.header("Backtest")
        selected_label = st.selectbox("Strategy", list(STRATEGIES.keys()), index=2)
        strategy = STRATEGIES[selected_label]["type"]
        initial_equity = st.number_input("Initial Equity", value=DEFAULT_INITIAL_EQUITY)
        size = st.text_input("Position %", value=DEFAULT_POSITION_SIZE)
        fees = st.number_input("Fees (as %)", value=DEFAULT_FEES, format="%.4f")
        backtest_clicked = st.button("Run Backtest")

    # Display price chart using historical data
    if historical_clicked:
        ticker_data = fetch_historical_data(ticker)
        if ticker_data is not None:
            price_fig = plot_price_chart(ticker_data, ticker)
            st.plotly_chart(price_fig, use_container_width=True)

    # Run backtest and display equity curve and additional charts
    if backtest_clicked:
        data = fetch_historical_data(ticker)
        if data is not None:
            entries, exits = get_strategy_signals(data, strategy)
            size_value = float(size) / 100.0
            portfolio = run_backtest(data, entries, exits, size_value, fees, initial_equity)

            # Display price chart
            price_fig = plot_price_chart(data, ticker)
            st.plotly_chart(price_fig, use_container_width=True)

            # Display equity curve for the ticker
            equity_fig = plot_equity_curve(portfolio.value(), strategy)
            st.plotly_chart(equity_fig, use_container_width=True)

            # Drawdown chart
            drawdown_trace = go.Scatter(
                x=(portfolio.drawdown() * 100).index,
                y=(portfolio.drawdown() * 100),
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red')
            )
            drawdown_fig = go.Figure(data=[drawdown_trace])
            drawdown_fig.update_layout(
                title='Drawdown',
                xaxis_title='Date',
                yaxis_title='% Drawdown',
                template='plotly_white'
            )
            st.plotly_chart(drawdown_fig, use_container_width=True)

            st.markdown("**Portfolio Plot:**")
            st.plotly_chart(portfolio.plot(), use_container_width=True)

if __name__ == "__main__":
    main()
