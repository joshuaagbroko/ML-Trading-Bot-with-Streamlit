import pandas as pd
import numpy as np
import ta  # Technical Analysis library

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # === Lagged returns ===
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)

    # === Moving Averages ===
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # === Volatility ===
    df['volatility_10'] = df['return_1d'].rolling(window=10).std()

    # === RSI (from `ta` lib) ===
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

    # === Volume change ===
    df['volume_change'] = df['volume'].pct_change()

    # Drop NaNs caused by rolling indicators
    df.dropna(inplace=True)

    return df


def label_data(df: pd.DataFrame, forward_days: int = 5, threshold: float = 0.02) -> pd.DataFrame:
    """
    Adds a binary 'target' column based on future return over a given number of days.
    Label = 1 if forward return > threshold, else 0

    :param df: Feature-engineered DataFrame with a 'close' column
    :param forward_days: Days ahead for calculating return
    :param threshold: Return threshold to label as 1 (buy signal)
    :return: DataFrame with 'target' column
    """
    df = df.copy()
    df['future_return'] = df['close'].shift(-forward_days) / df['close'] - 1
    df['target'] = (df['future_return'] > threshold).astype(int)
    df.dropna(inplace=True)
    return df

