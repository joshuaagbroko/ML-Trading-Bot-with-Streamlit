import sys
import os

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import warnings

warnings.filterwarnings("ignore")

from data_fetcher import get_price_data
from feature_engineer import add_features, label_data
from backtester import backtest_strategy

# Streamlit settings
st.set_page_config(page_title="ML Trading Strategy", layout="wide")
st.title("📈 ML-Powered Trading Strategy Dashboard")

# Sidebar input
ticker = st.sidebar.text_input("Enter Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
threshold = st.sidebar.slider("Labeling Threshold (Return %)", 0.0, 0.10, 0.02, step=0.005)

run_button = st.sidebar.button("🚀 Run Strategy")

if run_button:
    try:
        st.subheader(f"Fetching data for {ticker}...")
        df_raw = get_price_data(ticker, start=start_date.strftime("%Y-%m-%d"))

        st.success("✅ Price data loaded")

        st.subheader("🔧 Generating Features")
        df_feat = add_features(df_raw)

        st.subheader("🏷️ Labeling Data")
        df_labeled = label_data(df_feat, forward_days=5, threshold=threshold)

        # Feature columns used during training
        features = ["return_1d", "return_5d", "sma_10", "sma_50", "rsi_14", "volatility_10", "volume_change"]

        # Load trained model
        st.subheader("🤖 Loading Model and Making Predictions")
        model = load("trained_model.joblib")
        df_labeled["prediction"] = model.predict(df_labeled[features])

        st.subheader("🔁 Backtesting Strategy")
        df_bt, metrics = backtest_strategy(df_labeled)

        # Plot capital vs benchmark
        st.subheader("💹 Strategy vs. Benchmark")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_bt[["capital", "benchmark"]].plot(ax=ax)
        ax.set_title("Strategy Equity Curve vs. Buy-and-Hold")
        st.pyplot(fig)

        # Show performance metrics
        st.subheader("📊 Performance Metrics")
        st.write(metrics)

        # Optional raw data
        with st.expander("🔍 Show Data Table"):
            st.dataframe(df_bt.tail(20))

    except Exception as e:
        st.error(f"❌ Error: {e}")
