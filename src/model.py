# src/model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_rf_model(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100, random_state: int = 42) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor on the provided training data.
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MSE and MAE.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {"mse": mse, "mae": mae}

def reconstruct_prices(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Compute predicted future prices using:
      Predicted_Price = Current_Close * (1 + Predicted_Return)
    Also, create the Actual_Future_Price column from the data.
    """
    df["Predicted_Price"] = df["Close"] * (1 + df["Predicted_Return"])
    df["Actual_Future_Price"] = df["Close"].shift(-horizon)
    df.dropna(subset=["Predicted_Price", "Actual_Future_Price"], inplace=True)
    return df
