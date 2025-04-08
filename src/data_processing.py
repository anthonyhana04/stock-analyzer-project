# src/data_processing.py
import pandas as pd
import numpy as np

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns (if any) to a simple Index."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a 7-day moving average, daily returns, and add them to the DataFrame.
    """
    df["MA_7"] = df["Close"].rolling(window=7).mean()
    df["Return"] = df["Close"].pct_change()
    return df

def prepare_data_for_return_prediction(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Compute the target cumulative return for a given horizon.
    Target_Return = (Close[t+horizon] / Close[t]) - 1.
    Drop rows with missing values.
    """
    df["Target_Return"] = df["Close"].shift(-horizon) / df["Close"] - 1
    df.dropna(inplace=True)
    return df

# You can add other helper functions here if needed.
