import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Chart - Setup Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Chart")

# Chart Settings
st.header("Chart Settings")

# Arrange controls in a single row
col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1])

with col1:
    # Symbol input (default to S&P 500)
    symbol = st.text_input(
        "Symbol",
        value="^GSPC",
        help="Enter stock symbol (e.g., ^GSPC for S&P 500, AAPL for Apple)"
    )

with col2:
    # Period selector
    period = st.selectbox(
        "Period",
        options=["1d", "5d", "1mo", "3mo", "6mo",
                 "1y", "2y", "5y", "10y", "ytd", "max"],
        index=5,  # Default to "1y"
        help="Select the time period for historical data"
    )

with col3:
    # Interval selector
    interval = st.selectbox(
        "Interval",
        options=[ "1h", "1d", "5d", "1wk", "1mo", "3mo"],
        index=1,  # Default to "1d"
        help="Select the data interval (1m, 5m, 1h, 1d, 1wk, 1mo, etc.)"
    )

with col4:
    # Plot button
    plot_button = st.button(
        "📈 Plot Chart",
        type="primary",
        use_container_width=True
    )

# Link to view available symbols
st.markdown(
    "[🔗 View available symbols](https://finance.yahoo.com/markets/world-indices/)"
)
st.markdown("---")

# Main content area
if plot_button:
    try:
        with st.spinner("Downloading data..."):
            # Download historical data with selected period and interval
            data = yf.download(symbol, period=period, interval=interval)

            if data.empty:
                st.error(f"No data found for symbol: {symbol}")
            else:
                # Extract OHLC data
                # Handle both single ticker and multi-ticker data structures
                if isinstance(data.columns, pd.MultiIndex):
                    ohlc_data = data[['Open', 'High', 'Low', 'Close']]
                    open_col = ohlc_data[('Open', symbol)]
                    high_col = ohlc_data[('High', symbol)]
                    low_col = ohlc_data[('Low', symbol)]
                    close_col = ohlc_data[('Close', symbol)]
                else:
                    ohlc_data = data[['Open', 'High', 'Low', 'Close']]
                    open_col = ohlc_data['Open']
                    high_col = ohlc_data['High']
                    low_col = ohlc_data['Low']
                    close_col = ohlc_data['Close']

                # Create candlestick chart
                fig = go.Figure(data=[go.Candlestick(
                    x=ohlc_data.index,
                    open=open_col,
                    high=high_col,
                    low=low_col,
                    close=close_col,
                    name=symbol
                )])

                # Update layout
                fig.update_layout(
                    title=f"{symbol} Candlestick Chart ({period}, {interval})",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    height=600
                )

                # Display the chart
                st.plotly_chart(fig, width='stretch')

                # Show some basic stats
                st.subheader("📊 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${close_col.iloc[-1]:.2f}")
                with col2:
                    st.metric("High", f"${high_col.max():.2f}")
                with col3:
                    st.metric("Low", f"${low_col.min():.2f}")
                with col4:
                    change = close_col.iloc[-1] - close_col.iloc[0]
                    change_pct = (change / close_col.iloc[0]) * 100
                    st.metric("Change", f"${change:.2f}", f"{change_pct:.2f}%")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check that the symbol is valid and try again.")
else:
    st.info(
        "👈 Enter a symbol above, then click 'Plot Chart' to visualize the data.")
