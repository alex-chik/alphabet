import streamlit as st
import vectorbt as vbt
import pandas as pd
import pytz
import plotly.graph_objs as go
from datetime import datetime, date
from typing import Tuple, Optional

st.set_page_config(page_title="Stocks", layout="wide")

# Constants
DEFAULT_TICKER = "NFLX"
DEFAULT_START_DATE = date(2008, 1, 1)
DEFAULT_END_DATE = date(2008, 12, 31)
DEFAULT_INITIAL_EQUITY = 100000
DEFAULT_POSITION_SIZE = "50"
DEFAULT_FEES = 0.12

# Updated strategies dictionary for longer-term stock investing
STRATEGIES = {
    "SMA (50 & 200)": {"type": "SMA"},
    "EMA (20 & 50)": {"type": "EMA"},
    "RSI (14, 30/70)": {"type": "RSI"},
}

def convert_to_timezone_aware(date_obj: date) -> datetime:
    """Convert date to timezone-aware datetime."""
    return datetime.combine(date_obj, datetime.min.time()).astimezone(pytz.UTC)

@st.cache_data
def fetch_historical_data(ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.Series]:
    """Fetch and validate historical price data."""
    try:
        data = vbt.YFData.download(ticker, start=start_date, end=end_date).get('Close')
        if data is None or data.empty:
            st.error(f"No data available for {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def normalize_price_data(data: pd.Series) -> pd.Series:
    """Normalize price data to start at 100."""
    return data / data.iloc[0] * 100

def plot_price_comparison(ticker_data: pd.Series, spy_data: pd.Series, ticker: str) -> go.Figure:
    """Create price comparison chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spy_data.index, y=spy_data.values, mode='lines', name="SPY"))
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data.values, mode='lines', name=ticker))
    
    fig.update_layout(
        title=f"{ticker} vs SPY Normalized Stock Price",
        xaxis_title="Date",
        yaxis_title="Normalized Price"
    )
    return fig

def get_strategy_signals(data: pd.Series, strategy: str):
    """
    Generate entry and exit signals for longer-term stock investing.
    
    Parameters:
        data (pd.Series): Price data (daily closing prices are common).
        strategy (str): One of "SMA", "EMA", or "RSI".
        
    Returns:
        Tuple[pd.Series, pd.Series]: Boolean series for entries and exits.
    """
    if strategy == "SMA":
        # Use a 50-day SMA and a 200-day SMA
        fast = vbt.MA.run(data, window=50)
        slow = vbt.MA.run(data, window=200)
        entries = fast.ma_crossed_above(slow)
        exits = fast.ma_crossed_below(slow)
    elif strategy == "EMA":
        # Use a 20-day EMA and a 50-day EMA
        fast = vbt.MA.run(data, window=20, ewm=True)
        slow = vbt.MA.run(data, window=50, ewm=True)
        entries = fast.ma_crossed_above(slow)
        exits = fast.ma_crossed_below(slow)
    elif strategy == "RSI":
        # Use a 14-day RSI; entry when RSI < 30, exit when RSI > 70
        rsi = vbt.RSI.run(data, window=14)
        entries = rsi.rsi_below(30)
        exits = rsi.rsi_above(70)
    else:
        raise ValueError("Unknown strategy type provided.")
    
    return entries, exits

def run_backtest(data: pd.Series, entries: pd.Series, exits: pd.Series, 
                 size: float, fees: float, initial_equity: float) -> vbt.Portfolio:
    """Execute backtest with given parameters."""
    return vbt.Portfolio.from_signals(
        data, entries, exits,
        direction='longonly',
        size=size,
        size_type='percent',
        fees=fees/100,
        init_cash=initial_equity,
        freq='1D',
        min_size=1,
        size_granularity=1
    )

def plot_equity_curves(portfolio_value: pd.Series, baseline_value: pd.Series, strategy: str) -> go.Figure:
    """Create equity curve comparison chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=baseline_value.index, y=baseline_value, mode='lines', name='SPY'))
    fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', name=strategy))
    
    fig.update_layout(
        title=f"{strategy} Equity Curve vs SPY Hold Baseline",
        xaxis_title="Date",
        yaxis_title="Equity"
    )
    return fig

def main():
    """Main application function."""
    charts = st.empty()

    # Initialize session state
    if "data_fetched" not in st.session_state:
        st.session_state.data_fetched = False

    # Sidebar controls
    with st.sidebar:
        st.header("Historical")
        ticker = st.text_input("Ticker", value=DEFAULT_TICKER)
        start_date = st.date_input("Start Date", value=DEFAULT_START_DATE)
        end_date = st.date_input("End Date", value=DEFAULT_END_DATE)
        historical_clicked = st.button("Get Historical")

        st.header("Backtest")
        selected_label = st.selectbox("Strategy", list(STRATEGIES.keys()), index=2)
        strategy = STRATEGIES[selected_label]["type"]
        initial_equity = st.number_input("Initial Equity", value=DEFAULT_INITIAL_EQUITY)
        size = st.text_input("Position %", value=DEFAULT_POSITION_SIZE)
        fees = st.number_input("Fees (as %)", value=DEFAULT_FEES, format="%.4f")
        backtest_clicked = st.button("Run Backtest")

    # Handle historical data
    if not st.session_state.data_fetched or historical_clicked:
        start_date_tz = convert_to_timezone_aware(start_date)
        end_date_tz = convert_to_timezone_aware(end_date)

        spy_data = fetch_historical_data("SPY", start_date_tz, end_date_tz)
        ticker_data = fetch_historical_data(ticker, start_date_tz, end_date_tz)

        if spy_data is not None and ticker_data is not None:
            spy_normalized = normalize_price_data(spy_data)
            ticker_normalized = normalize_price_data(ticker_data)
            
            st.session_state.data_fetched = True
            fig = plot_price_comparison(ticker_normalized, spy_normalized, ticker)
            st.plotly_chart(fig, use_container_width=True)

    # Handle backtest
    if backtest_clicked:
        start_date_tz = convert_to_timezone_aware(start_date)
        end_date_tz = convert_to_timezone_aware(end_date)

        # Fetch fresh data for backtest
        data = fetch_historical_data(ticker, start_date_tz, end_date_tz)
        spy_data = fetch_historical_data("SPY", start_date_tz, end_date_tz)

        if data is not None and spy_data is not None:
            # Generate signals and run portfolio
            entries, exits = get_strategy_signals(data, strategy)
            size_value = float(size) / 100.0
            
            portfolio = run_backtest(data, entries, exits, size_value, fees, initial_equity)
            baseline = vbt.Portfolio.from_holding(spy_data, init_cash=initial_equity)

            # Create and display charts
            data_normalized = normalize_price_data(data)
            spy_normalized = normalize_price_data(spy_data)
            
            price_fig = plot_price_comparison(data_normalized, spy_normalized, ticker)
            equity_fig = plot_equity_curves(portfolio.value(), baseline.value(), strategy)
            
            st.plotly_chart(price_fig, use_container_width=True)
            st.plotly_chart(equity_fig, use_container_width=True)

if __name__ == "__main__":
    main()


    