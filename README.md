# 🤖 Automated Trading Signal Generator with ML & Streamlit

This project is a complete machine learning-driven trading strategy powered by [OpenBB](https://openbb.co/), Streamlit, and scikit-learn. It uses historical stock data to generate trading signals based on technical indicators, trains an ML model, backtests the strategy, and visualizes everything in an interactive dashboard.

## 🚀 Features

- 📥 Real-time price data fetching via OpenBB
- 🧠 ML model training using Random Forest
- 🔧 Technical feature engineering (RSI, SMA, volatility, etc.)
- 🏷️ Custom labeling based on forward returns
- 📈 Strategy backtesting with equity curve vs. benchmark
- 📊 Performance metrics (Total Return, Sharpe Ratio, etc.)
- 🖥️ Interactive dashboard built with Streamlit

## 📁 Project Structure

Automated Trading Bot/
-├── app/
-│ ├── dashboard.py # Streamlit dashboard
-│ ├── data_fetcher.py # Data via OpenBB
-│ ├── feature_engineer.py # Technical indicators + labels
-│ ├── backtester.py # Strategy simulation logic
-│ └── model_trainer.py # Model training script
-├── trained_model.joblib # Saved ML model
-└── README.md

## 📦 Dependencies
- streamlit
- scikit-learn
- pandas, numpy
- matplotlib
- ta (Technical Analysis)
- openbb (OpenBB SDK)
- joblib

## 🧠 Author
Built by Joshua Agbroko
