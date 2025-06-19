from openbb import obb
import pandas as pd

def get_price_data(ticker: str, start: str = "2022-01-01", interval: str = "1d") -> pd.DataFrame:
    print("✅ Called get_price_data")  # <== Diagnostic

    response = obb.equity.price.historical(
        symbol=ticker,
        start_date=start,
        interval=interval,
    )

    print("⚙️  Response type:", type(response))  # <== Should print OBBject

    try:
        df = response.to_dataframe()
        print("📊 Converted to DataFrame, shape:", df.shape)
    except Exception as e:
        raise ValueError(f"Failed to convert to DataFrame: {e}")

    if df.empty:
        raise ValueError(f"No data for {ticker}")

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

