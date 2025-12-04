import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from lib.line import Calculator

st.set_page_config(
    page_title="Chart - Setup Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Chart")

# Chart Settings
st.header("Chart Settings")

# Arrange controls in a single row
col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])

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
        options=["1h", "1d", "5d", "1wk", "1mo", "3mo"],
        index=1,  # Default to "1d"
        help="Select the data interval (1m, 5m, 1h, 1d, 1wk, 1mo, etc.)"
    )

with col4:
    # Chart type selector
    chart_type = st.selectbox(
        "Chart Type",
        options=["Candlestick", "Bar"],
        index=1,  # Default to "Bar" as per recent change
        help="Select the chart style"
    )

with col5:
    # Index selector for resistance/support calculations
    bar_index = st.number_input(
        "Bar Index",
        value=-1,
        min_value=-1000,
        max_value=1000,
        step=1,
        help="Index of the bar to calculate lines for (-1 = last bar, -2 = second to last, etc.)"
    )

with col6:
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

# Initialize session state
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None
if 'chart_fig' not in st.session_state:
    st.session_state.chart_fig = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

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

                # Calculate dots using Calculator (includes x_position calculation)
                calculator = Calculator()
                dots_valid = calculator.calculate_dots(data, symbol)

                # Store data in session state (don't store calculator, recreate it when needed)
                st.session_state.chart_data = {
                    'data': data,
                    'dots_valid': dots_valid,
                    'symbol': symbol,
                    'ohlc_data': ohlc_data,
                    'open_col': open_col,
                    'high_col': high_col,
                    'low_col': low_col,
                    'close_col': close_col,
                    'chart_type': chart_type,
                    'period': period,
                    'interval': interval
                }

                # Create chart based on selection
                if chart_type == "Bar":
                    trace = go.Ohlc(
                        x=ohlc_data.index,
                        open=open_col,
                        high=high_col,
                        low=low_col,
                        close=close_col,
                        name=symbol,
                    )
                else:
                    trace = go.Candlestick(
                        x=ohlc_data.index,
                        open=open_col,
                        high=high_col,
                        low=low_col,
                        close=close_col,
                        name=symbol,
                    )
                fig = go.Figure(data=[trace])

                # Add dots scatter plot if we have valid dots
                if len(dots_valid) > 0:
                    fig.add_trace(go.Scatter(
                        x=dots_valid['x_position'],
                        y=dots_valid['dots'],
                        mode='markers',
                        name='Dots',
                        marker=dict(
                            size=8,
                            color='white',
                            symbol='circle',
                            line=dict(width=1, color='darkred')
                        )
                    ))

                # Define line functions configuration
                line_configs = [
                    {
                        'func': calculator.get_5_2_resistance,
                        'args': lambda: (data, symbol, bar_index),
                        'name': '5/2 Resistance',
                        'text': lambda p1, p2: f"5-2D {p2[1]:.2f}",
                        'color': 'red',
                        'min_bars': 3
                    },
                    {
                        'func': calculator.get_5_2_support,
                        'args': lambda: (data, symbol, bar_index),
                        'name': '5/2 Support',
                        'text': lambda p1, p2: f"5-2U {p2[1]:.2f}",
                        'color': 'green',
                        'min_bars': 3
                    },
                    {
                        'func': calculator.get_5_1_resistance,
                        'args': lambda: (data, symbol, bar_index),
                        'name': '5/1 Resistance',
                        'text': lambda p1, p2: f"5-1D {p2[1]:.2f}",
                        'color': 'red',
                        'min_bars': 3
                    },
                    {
                        'func': calculator.get_5_1_support,
                        'args': lambda: (data, symbol, bar_index),
                        'name': '5/1 Support',
                        'text': lambda p1, p2: f"5-1U {p2[1]:.2f}",
                        'color': 'green',
                        'min_bars': 3
                    },
                    {
                        'func': calculator.get_5_3_support,
                        'args': lambda: (data, symbol, bar_index),
                        'name': '5/3 Support',
                        'text': lambda p1, p2: f"5-3U {p2[1]:.2f}",
                        'color': 'green',
                        'min_bars': 3
                    },
                    {
                        'func': calculator.get_5_3_resistance,
                        'args': lambda: (data, symbol, bar_index),
                        'name': '5/3 Resistance',
                        'text': lambda p1, p2: f"5-3D {p2[1]:.2f}",
                        'color': 'red',
                        'min_bars': 3
                    },
                    {
                        'func': calculator.get_5_9_support,
                        'args': lambda: (data, symbol, bar_index),
                        'name': '5/9 Support',
                        'text': lambda p1, p2: f"5-9U {p2[1]:.2f}",
                        'color': 'green',
                        'min_bars': 3
                    },
                    {
                        'func': calculator.get_5_9_resistance,
                        'args': lambda: (data, symbol, bar_index),
                        'name': '5/9 Resistance',
                        'text': lambda p1, p2: f"5-9D {p2[1]:.2f}",
                        'color': 'red',
                        'min_bars': 3
                    },
                    {
                        'func': calculator.get_6_1_support,
                        'args': lambda: (data, dots_valid['dots'], symbol, bar_index),
                        'name': '6/1 Support',
                        'text': lambda p1, p2: f"6-1U {p2[1]:.2f}",
                        'color': 'green',
                        'min_bars': 3
                    },
                    {
                        'func': calculator.get_6_1_resistance,
                        'args': lambda: (data, dots_valid['dots'], symbol, bar_index),
                        'name': '6/1 Resistance',
                        'text': lambda p1, p2: f"6-1D {p2[1]:.2f}",
                        'color': 'red',
                        'min_bars': 3
                    },
                    {
                        'func': calculator.get_6_5_resistance,
                        'args': lambda: (data, dots_valid['dots'], symbol, bar_index),
                        'name': '6/5 Resistance',
                        'text': lambda p1, p2: f"6-5D {p2[1]:.2f}",
                        'color': 'red',
                        'min_bars': 3
                    },
                    {
                        'func': calculator.get_6_5_support,
                        'args': lambda: (data, dots_valid['dots'], symbol, bar_index),
                        'name': '6/5 Support',
                        'text': lambda p1, p2: f"6-5U {p2[1]:.2f}",
                        'color': 'green',
                        'min_bars': 3
                    },
                    {
                        'func': calculator.get_6_7_support,
                        'args': lambda: (data, dots_valid['dots'], symbol, bar_index),
                        'name': '6/7 Support',
                        'text': lambda p1, p2: f"6-7U {p2[1]:.2f}",
                        'color': 'green',
                        'min_bars': 3
                    },
                    {
                        'func': calculator.get_6_7_resistance,
                        'args': lambda: (data, dots_valid['dots'], symbol, bar_index),
                        'name': '6/7 Resistance',
                        'text': lambda p1, p2: f"6-7D {p2[1]:.2f}",
                        'color': 'red',
                        'min_bars': 3
                    },
                ]

                # Collect resistance and support data for LLM analysis
                resistances = []
                supports = []

                # Add lines from configuration
                for config in line_configs:
                    if len(ohlc_data) >= config['min_bars']:
                        args = config['args']()  # Call lambda to get args
                        point_1, point_2 = config['func'](*args)
                        if point_1 is not None and point_2 is not None:
                            # Collect data for analysis
                            if 'Resistance' in config['name']:
                                resistances.append((point_1, point_2))
                            elif 'Support' in config['name']:
                                supports.append((point_1, point_2))

                            # Add the line
                            fig.add_trace(go.Scatter(
                                x=[point_1[0], point_2[0]],
                                y=[point_1[1], point_2[1]],
                                mode='lines',
                                name=config['name'],
                                line=dict(
                                    width=2,
                                    color=config['color'],
                                    dash='solid'
                                ),
                                showlegend=False
                            ))
                            # Add markers with text labels
                            text_value = config['text'](point_1, point_2)
                            fig.add_trace(go.Scatter(
                                x=[point_2[0]],
                                y=[point_2[1]],
                                mode='markers+text',
                                name=f'{config["name"]} Points',
                                text=[text_value],
                                textposition="top center",
                                marker=dict(
                                    size=10,
                                    color=config['color'],
                                    symbol='circle'
                                ),
                                textfont=dict(
                                    size=12,
                                    color=config['color']
                                ),
                                showlegend=False
                            ))

                # Update layout with auto-scale initially
                fig.update_layout(
                    title=f"{symbol} {chart_type} Chart ({period}, {interval})",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    height=700,  # Good height for the chart
                    dragmode="pan",  # Set pan as the default tool
                    autosize=True,
                )

                # After layout, zoom to show last n bars
                n = 5
                num_candles = len(ohlc_data)
                if num_candles > n:
                    # Get the last n index values for x-axis
                    last_n_indices = ohlc_data.index[-n:]
                    x_range_start = last_n_indices[0]
                    x_range_end = last_n_indices[-1]

                    # Get the last n candles for y-axis range calculation
                    last_n_high = high_col.iloc[-n:].max()
                    last_n_low = low_col.iloc[-n:].min()

                    # Include resistance and support points that intersect with the visible range
                    all_y_values = [last_n_high, last_n_low]

                    # Check all resistance and support lines
                    for line_points in resistances + supports:
                        point_1, point_2 = line_points
                        _, y1 = point_1  # Only use y value, ignore timestamp index
                        _, y2 = point_2  # Only use y value, ignore timestamp index

                        # Check if line intersects with visible x-axis range using timestamps
                        x1, _ = point_1
                        x2, _ = point_2
                        line_x_min = min(x1, x2)
                        line_x_max = max(x1, x2)

                        # If line overlaps with visible range, add its y values
                        if line_x_max >= x_range_start and line_x_min <= x_range_end:
                            all_y_values.extend([y1, y2])

                    # Calculate min and max including resistance/support points
                    y_min = min(all_y_values)
                    y_max = max(all_y_values)

                    # Add some padding (5% on each side)
                    y_padding = (y_max - y_min) * 0.1
                    y_range = [y_min - y_padding, y_max + y_padding]

                    fig.update_xaxes(
                        range=[x_range_start, x_range_end])
                    # Set y-axis to fit the last n candles and visible resistance/support points
                    fig.update_yaxes(range=y_range)
                else:
                    # If there are n or fewer bars, show all (already auto-scaled)
                    pass

                # Store figure in session state
                st.session_state.chart_fig = fig
                st.session_state.chart_data['resistances'] = resistances
                st.session_state.chart_data['supports'] = supports

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check that the symbol is valid and try again.")
        st.session_state.chart_data = None
        st.session_state.chart_fig = None

# Display chart if available in session state
if st.session_state.chart_fig is not None and st.session_state.chart_data is not None:
    chart_data = st.session_state.chart_data

    # Display the chart
    st.plotly_chart(st.session_state.chart_fig, width='stretch', config={
                    "modeBarButtonsToAdd": ["pan2d"]})

    # Show some basic stats
    st.subheader("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${chart_data['close_col'].iloc[-1]:.2f}")
    with col2:
        st.metric("High", f"${chart_data['high_col'].max():.2f}")
    with col3:
        st.metric("Low", f"${chart_data['low_col'].min():.2f}")
    with col4:
        change = chart_data['close_col'].iloc[-1] - \
            chart_data['close_col'].iloc[0]
        change_pct = (change / chart_data['close_col'].iloc[0]) * 100
        st.metric("Change", f"${change:.2f}", f"{change_pct:.2f}%")

    # LLM Analysis Section
    st.markdown("---")
    st.subheader("🤖 LLM Analysis")

    analyze_button = st.button(
        "🔍 Analyze using LLM",
        type="primary",
        use_container_width=True
    )

    if analyze_button or st.session_state.analysis_result is not None:
        if analyze_button:
            with st.spinner("Analyzing with LLM..."):
                try:
                    # Get API key from Streamlit secrets
                    api_key = None
                    try:
                        api_key = st.secrets.api_key
                    except:
                        st.error("API key not found")

                    # Recreate calculator
                    calculator = Calculator()
                    # Get analysis from Groq
                    analysis_result = calculator.get_gpt_analysis(
                        chart_data['data'],
                        chart_data['dots_valid'],
                        chart_data['symbol'],
                        chart_data['resistances'],
                        chart_data['supports'],
                        api_key=api_key
                    )
                    # Store in session state
                    st.session_state.analysis_result = analysis_result
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.session_state.analysis_result = None

        # Display stored analysis result
        if st.session_state.analysis_result is not None:
            analysis_result = st.session_state.analysis_result

            if analysis_result.get('error'):
                st.error(f"Error: {analysis_result['error']}")
            else:
                # Display analysis
                if analysis_result.get('analysis'):
                    st.markdown("### 📝 Analysis")
                    st.write(analysis_result['analysis'])

                # Display probabilities
                col_prob1, col_prob2 = st.columns(2)
                with col_prob1:
                    prob_up = analysis_result.get('probability_up')
                    if prob_up is not None:
                        st.metric(
                            "Probability UP",
                            f"{prob_up:.1f}%",
                            delta=f"{prob_up:.1f}%"
                        )

                with col_prob2:
                    prob_down = analysis_result.get('probability_down')
                    if prob_down is not None:
                        st.metric(
                            "Probability DOWN",
                            f"{prob_down:.1f}%",
                            delta=f"-{prob_down:.1f}%"
                        )

                # Show raw response in expander
                with st.expander("View Raw Response"):
                    st.text(analysis_result.get('raw_response', 'No response'))

else:
    if st.session_state.chart_data is None:
        st.info(
            "Enter a symbol above, then click 'Plot Chart' to visualize the data.")
