import streamlit as st
import vectorbt as vbt
import pandas as pd
import pytz
import plotly.graph_objs as go
from datetime import datetime, date

charts = st.empty()

# Convert date to timezone-aware datetime
def convert_to_timezone_aware(date_obj):
    return datetime.combine(date_obj, datetime.min.time()).astimezone(pytz.UTC)

# Initialize session state for first run
if "data_fetched" not in st.session_state:
    st.session_state.data_fetched = False

with st.sidebar:
    st.header("Historical")

    ticker = st.text_input("Ticker", value="NFLX")
    start_date = st.date_input("Start Date", value=date(2008, 1, 1))
    end_date = st.date_input("End Date", value=date(2008, 12, 31))

    # Button to manually refresh data
    historical_clicked = st.button("Get Historical")

    st.header("Backtest")

    # Backtesting controls
    strategies = {
        "SMA (50 & 200)": {"type": "SMA"},
        "EMA (50 & 200)": {"type": "EMA"},
        "RSI (70)": {"type": "RSI"},
    }

    # User selects a strategy
    selected_label = st.selectbox("Strategy", list(strategies.keys()), index=2)

    # Get corresponding internal values
    strategy = strategies[selected_label]["type"]
    initial_equity = st.number_input("Initial Equity", value=100000)
    size = st.text_input("Position %", value='50')
    fees = st.number_input("Fees (as %)", value=0.12, format="%.4f")

    # Button to perform backtesting
    backtest_clicked = st.button("Run Backtest")

# Run once on startup or when button is clicked
if not st.session_state.data_fetched or historical_clicked:
    try:     
        start_date_tz = convert_to_timezone_aware(start_date)
        end_date_tz = convert_to_timezone_aware(end_date)

        # Fetch SPY data
        spy_data = vbt.YFData.download("SPY", start=start_date_tz, end=end_date_tz).get('Close')
        if spy_data is None or spy_data.empty:
            st.error("Failed to fetch SPY data. Please check the date range.")
            st.stop()
        
        spy_data = spy_data / spy_data.iloc[0] * 100  # Normalize

        # Fetch ticker data
        data = vbt.YFData.download(ticker, start=start_date_tz, end=end_date_tz).get('Close')
        if data is None or data.empty:
            st.error(f"Failed to fetch data for {ticker}. Please check the ticker symbol.")
            st.stop()

        data = data / data.iloc[0] * 100  # Normalize

        # Store data in session state
        st.session_state.data_fetched = True

        # Plotly Figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spy_data.index, y=spy_data.values, mode='lines', name="SPY"))
        fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', name=ticker))

        fig.update_layout(title=f"{ticker} vs SPY Normalized Stock Price", 
                          xaxis_title="Date", 
                          yaxis_title="Normalized Price")
  
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Run backtest when button is clicked
if backtest_clicked:
    
    short, long, entries, exits = None, None, None, None
    start_date_tz = convert_to_timezone_aware(start_date)
    end_date_tz = convert_to_timezone_aware(end_date)

    # Fetch data
    spy_data = vbt.YFData.download("SPY", start=start_date_tz, end=end_date_tz).get('Close')
    spy_data_normalized = spy_data / spy_data.iloc[0] * 100  # Normalize
    data = vbt.YFData.download(ticker, start=start_date_tz, end=end_date_tz).get('Close')
    data_normalized = data / data.iloc[0] * 100  # Normalize

    if strategy == "SMA":
        short = vbt.MA.run(data, 50, short_name='fast')
        long = vbt.MA.run(data, 200, short_name='slow')
        entries = short.ma_crossed_above(long)
        exits = short.ma_crossed_below(long)
    elif strategy == "EMA":
        short = vbt.MA.run(data, 50, short_name='fast', ewm=True)
        long = vbt.MA.run(data, 200, short_name='slow', ewm=True)
        entries = short.ma_crossed_above(long)
        exits = short.ma_crossed_below(long)
    elif strategy == "RSI":
        rsi = vbt.RSI.run(data, 14)
        entries = rsi.rsi_below(30)
        exits = rsi.rsi_above(70)
 
    # Convert size to appropriate type
    size_value = float(size) / 100.0

    # Run portfolio
    portfolio = vbt.Portfolio.from_signals(
        data, entries, exits,
        direction='longonly',
        size=size_value,
        size_type='percent',
        fees=fees/100,
        init_cash=initial_equity,
        freq='1D',  
        min_size =1,
        size_granularity = 1
    )

    # Run baseline
    baseline = vbt.Portfolio.from_holding(spy_data, init_cash=initial_equity)
    
    # Plotting
    equity_data = portfolio.value()
    baseline_data = baseline.value()

    # Equity Curve
    e_fig = go.Figure()
    e_fig.add_trace(go.Scatter(x=baseline_data.index, y=baseline_data, mode='lines', name='SPY'))
    e_fig.add_trace(go.Scatter(x=equity_data.index, y=equity_data, mode='lines', name=strategy))
  
    e_fig.update_layout(title=f"{strategy} Equity Curve vs SPY Hold Baseline", 
                        xaxis_title="Date", yaxis_title="Equity")
    
    # Historical Data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spy_data_normalized.index, 
                             y=spy_data_normalized.values, mode='lines', name="SPY"))
    fig.add_trace(go.Scatter(x=data_normalized.index, 
                             y=data_normalized.values, mode='lines', name=ticker))

    fig.update_layout(title=f"{ticker} vs SPY Normalized Stock Price", 
                        xaxis_title="Date", 
                        yaxis_title="Normalized Price")
    
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(e_fig, use_container_width=True)


    