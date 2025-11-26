import streamlit as st

st.set_page_config(
    page_title="Setup Predictor",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Setup Predictor")

st.markdown("""
## Welcome to Setup Predictor

This application provides tools for analyzing and visualizing stock market data. 

### Features

- **Interactive Charts**: View candlestick charts for any stock symbol
- **Real-time Data**: Access up-to-date market data using Yahoo Finance
- **Easy Navigation**: Use the sidebar to navigate between different pages

### Getting Started

Navigate to the **Chart** page using the sidebar to start visualizing stock market data. 
Simply enter a stock symbol (like ^GSPC for S&P 500 or AAPL for Apple) and click the plot button to generate an interactive candlestick chart.

### Supported Symbols

You can use any valid stock ticker symbol, including:
- **^GSPC** - S&P 500 Index
- **AAPL** - Apple Inc.
- **MSFT** - Microsoft Corporation
- **GOOGL** - Alphabet Inc.
- And many more...

Enjoy exploring the markets! 📈
""")
