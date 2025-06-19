import pandas as pd
import numpy as np

def backtest_strategy(df: pd.DataFrame, signal_col: str = "prediction", starting_capital: float = 100_000):
    df = df.copy()

    df["returns"] = df["close"].pct_change().fillna(0)
    df["position"] = df[signal_col].shift(1).fillna(0)

    df["strategy_returns"] = df["position"] * df["returns"]
    df["capital"] = (1 + df["strategy_returns"]).cumprod() * starting_capital
    df["benchmark"] = (1 + df["returns"]).cumprod() * starting_capital

    # Calculate key metrics
    total_return = df["capital"].iloc[-1] / starting_capital - 1
    benchmark_return = df["benchmark"].iloc[-1] / starting_capital - 1
    excess_return = total_return - benchmark_return
    volatility = df["strategy_returns"].std() * np.sqrt(252)
    sharpe = df["strategy_returns"].mean() / df["strategy_returns"].std() * np.sqrt(252)

    metrics = {
        "Total Return": f"{total_return:.2%}",
        "Benchmark Return": f"{benchmark_return:.2%}",
        "Excess Return": f"{excess_return:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}"
    }

    return df, metrics
