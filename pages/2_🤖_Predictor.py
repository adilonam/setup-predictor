import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lib.line import Calculator
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Price Predictor - Setup Predictor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Price Predictor")

# Settings Section
st.header("Settings")

# Arrange controls in a single row
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

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
        help="Select the data interval"
    )

with col4:
    # Predict button
    predict_button = st.button(
        "🔮 Predict",
        type="primary",
        use_container_width=True
    )

st.markdown("---")

# Initialize session state
if 'predictor_data' not in st.session_state:
    st.session_state.predictor_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'imputer' not in st.session_state:
    st.session_state.imputer = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
# Main processing
if predict_button:
    try:
        # Initialize calculator with symbol, period, and interval
        calculator = Calculator(
            symbol=symbol, period=period, interval=interval)
        st.session_state.calculator = calculator

        with st.spinner("Downloading data..."):
            data = calculator.download_data()

            if data.empty:
                st.error(f"No data found for symbol: {symbol}")
            else:

                with st.spinner("Calculating resistance and support levels..."):
                    result_df = calculator.create_result_df(data)

                with st.spinner("Preparing data for model..."):
                    X_imputed, y, imputer = calculator.prepare_model_data(
                        result_df)

                    # Convert to numpy arrays
                    X_array = X_imputed.values.astype(np.float32)
                    y_array = y.values.astype(np.float32)

                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_array)

                    # Store scaler and imputer
                    st.session_state.scaler = scaler
                    st.session_state.imputer = imputer

                with st.spinner("Training model (this may take a moment)..."):
                    # Create and train model
                    n_features = X_scaled.shape[1]
                    model = calculator.create_model(n_features)

                    # Train model (using all data for simplicity, or split if needed)
                    # For faster predictions, we'll use a smaller number of epochs
                    history = model.fit(
                        X_scaled, y_array,
                        epochs=200,
                        batch_size=32,
                        verbose=0,
                        validation_split=0.2
                    )

                    st.session_state.model = model
                    st.session_state.training_history = history
                    st.session_state.predictor_data = {
                        'result_df': result_df,
                        'X_imputed': X_imputed,
                        'y': y,
                        'symbol': symbol,
                        'data': data
                    }

                st.success("Model trained and ready for predictions!")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# Display predictions if model is ready
if st.session_state.predictor_data is not None and st.session_state.model is not None:
    predictor_data = st.session_state.predictor_data
    model = st.session_state.model
    scaler = st.session_state.scaler
    imputer = st.session_state.imputer

    result_df = predictor_data['result_df']
    X_imputed = predictor_data['X_imputed']
    symbol = predictor_data['symbol']
    data = predictor_data['data']

    st.markdown("---")
    st.header("📊 Prediction Results")

    # Index selector for prediction
    max_idx = len(result_df) - 1
    st.subheader("Select Index for Prediction")
    col_idx1, col_idx2 = st.columns([3, 1])

    with col_idx1:
        selected_idx = st.slider(
            "Data Point Index",
            min_value=0,
            max_value=max_idx,
            value=max_idx,  # Default to last index
            help=f"Select which data point to predict (0 = oldest, {max_idx} = most recent)"
        )

    with col_idx2:
        if isinstance(result_df.index, pd.DatetimeIndex):
            selected_date = result_df.index[selected_idx]
            date_str = selected_date.strftime(
                '%Y-%m-%d') if hasattr(selected_date, 'strftime') else str(selected_date)
        else:
            date_str = str(result_df.index[selected_idx])
        st.metric("Selected Date", date_str)

    # Prepare selected row for prediction
    if isinstance(result_df.columns, pd.MultiIndex):
        feature_cols = [
            col for col in result_df.columns if col[0] != 'price_up']
        selected_row_features = result_df[feature_cols].iloc[selected_idx:selected_idx+1].copy()
    else:
        feature_cols = [col for col in result_df.columns if col != 'price_up']
        selected_row_features = result_df[feature_cols].iloc[selected_idx:selected_idx+1].copy()

    # Impute and scale
    selected_row_imputed = pd.DataFrame(
        imputer.transform(selected_row_features),
        columns=selected_row_features.columns,
        index=selected_row_features.index
    )
    selected_row_scaled = scaler.transform(
        selected_row_imputed.values.astype(np.float32))

    # Make prediction
    prediction_prob = model.predict(selected_row_scaled, verbose=0)[0][0]

    prediction_binary = 1 if prediction_prob > 0.5 else 0

    # Get price info for selected index
    if isinstance(data.columns, pd.MultiIndex):
        current_close = data[('Close', symbol)].iloc[selected_idx]
        current_date = data.index[selected_idx]
    else:
        current_close = data['Close'].iloc[selected_idx]
        current_date = data.index[selected_idx]

    # Display prediction with prominent styling
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 🎯 Current Prediction")

        # Probability display
        prob_percent = prediction_prob * 100

        if prediction_binary == 1:
            direction = "📈 UP"
            direction_text = "UP"
            color = "green"
            prob_color = "green"
        else:
            direction = "📉 DOWN"
            direction_text = "DOWN"
            color = "red"
            prob_color = "red"

        # Main metric display using HTML (dark mode compatible)
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: rgba(38, 39, 48, 0.8); border: 1px solid rgba(250, 250, 250, 0.2); border-radius: 10px; margin: 20px 0;">
            <h4 style="color: rgba(250, 250, 250, 0.9); margin-bottom: 10px;">Price Direction Prediction for {symbol}</h4>
            <h2 style="color: {color}; margin: 10px 0; font-size: 2.5em;">{direction}</h2>
            <p style="font-size: 1.2em; margin-top: 10px; color: rgba(250, 250, 250, 0.9);">
                Probability: <span style="color: {prob_color}; font-weight: bold; font-size: 1.3em;">{prob_percent:.4f}%</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Detailed probability (dark mode compatible)
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: rgba(38, 39, 48, 0.8); border: 1px solid rgba(250, 250, 250, 0.2); border-radius: 10px; margin: 20px 0;">
            <h3 style="color: {color}; margin-bottom: 10px;">Prediction Probability: {prob_percent:.4f}%</h3>
            <p style="font-size: 16px; color: rgba(250, 250, 250, 0.9);">
                <strong>Current Price:</strong> ${current_close:.2f}<br>
                <strong>Date:</strong> {current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date)}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Feature columns display (dark mode compatible)
        # Get feature values for the selected row (before imputation to show original values)
        feature_values_html = ""
        for col in feature_cols:
            col_name = col[0] if isinstance(col, tuple) else col
            value = selected_row_features[col].iloc[0]

            # Format the value based on its type
            if pd.isna(value):
                formatted_value = "N/A"
            elif isinstance(value, (int, float)):
                if abs(value) >= 1000000:
                    formatted_value = f"{value:,.0f}"
                elif abs(value) >= 1000:
                    formatted_value = f"{value:,.2f}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)

            feature_values_html += f'<div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(250, 250, 250, 0.1);"><span style="color: rgba(250, 250, 250, 0.7);">{col_name}:</span><span style="color: rgba(250, 250, 250, 0.9); font-weight: bold;">{formatted_value}</span></div>'

        st.markdown(f"""
        <div style="padding: 20px; background-color: rgba(38, 39, 48, 0.8); border: 1px solid rgba(250, 250, 250, 0.2); border-radius: 10px; margin: 20px 0;">
            <h3 style="color: rgba(250, 250, 250, 0.9); margin-bottom: 15px; text-align: center;">Feature Values for Selected Row</h3>
            <div style="max-height: 400px; overflow-y: auto;">
                {feature_values_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Prediction text
        if prob_percent >= 70:
            confidence = "Very High"
        elif prob_percent >= 60:
            confidence = "High"
        elif prob_percent >= 50:
            confidence = "Moderate"
        elif prob_percent >= 40:
            confidence = "Low"
        else:
            confidence = "Very Low"

        st.info(f"""
        **Prediction Summary:**
        - **Direction:** Price is predicted to go **{'UP' if prediction_binary == 1 else 'DOWN'}**
        - **Confidence:** {confidence} ({prob_percent:.2f}%)
        - **Raw Probability Value:** {prediction_prob:.6f}
        """)

    # Visualization
    st.markdown("---")
    st.subheader("📈 Price Chart with Prediction")

    # Create chart
    if isinstance(data.columns, pd.MultiIndex):
        close_col = data[('Close', symbol)]
        high_col = data[('High', symbol)]
        low_col = data[('Low', symbol)]
        open_col = data[('Open', symbol)]
    else:
        close_col = data['Close']
        high_col = data['High']
        low_col = data['Low']
        open_col = data['Open']

    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=open_col,
        high=high_col,
        low=low_col,
        close=close_col,
        name=symbol
    )])

    # Add prediction indicator at selected index
    prediction_color = 'green' if prediction_binary == 1 else 'red'
    prediction_marker = 'triangle-up' if prediction_binary == 1 else 'triangle-down'

    # Get the date for the selected index
    selected_date_for_chart = data.index[selected_idx]

    fig.add_trace(go.Scatter(
        x=[selected_date_for_chart],
        y=[current_close],
        mode='markers+text',
        name='Prediction',
        text=[f"{prob_percent:.1f}% {'UP' if prediction_binary == 1 else 'DOWN'}"],
        textposition="top center",
        marker=dict(
            size=20,
            color=prediction_color,
            symbol=prediction_marker,
            line=dict(width=2, color='white')
        ),
        showlegend=True
    ))

    fig.update_layout(
        title=f"{symbol} Price Chart with Prediction",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600,
        dragmode="pan"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Enter a symbol above and click 'Predict' to generate price predictions using the TensorFlow model.")
