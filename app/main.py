# app/main.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import yfinance as yf
import datetime
import numpy as np
import matplotlib.pyplot as plt

from src.data_processing import flatten_columns, add_technical_features, prepare_data_for_return_prediction
from src.model import train_rf_model, evaluate_model, reconstruct_prices

def main():
    st.title("Stock Analyzer with Random Forest")
    st.write("""
    This app fetches historical data for a stock via Yahoo Finance,
    trains a Random Forest model to predict cumulative returns over a specified horizon,
    and then computes price forecasts.
    """)

    ticker = st.text_input("Enter a ticker symbol (e.g., 'AAPL', 'NVDA', 'CLSK'):", "AAPL")
    end_date = datetime.date.today()
    start_date = st.date_input("Start Date:", datetime.date(end_date.year - 2, end_date.month, end_date.day))
    horizon = st.slider("Prediction Horizon (days):", min_value=1, max_value=365, value=1, step=1)

    if st.button("Fetch & Train Model"):
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            st.error("No data found for this ticker/date range.")
            return

        df = flatten_columns(df)
        df = add_technical_features(df)
        df = prepare_data_for_return_prediction(df, horizon)
        if df.empty or len(df) < 30:
            st.error("Not enough data points after processing. Try a longer date range.")
            return

        features = ["Close", "MA_7", "Volume", "Return"]
        target = "Target_Return"

        X = df[features].values
        y = df[target].values

        train_ratio = 0.8
        cutoff = int(len(df) * train_ratio)
        X_train, X_test = X[:cutoff], X[cutoff:]
        y_train, y_test = y[:cutoff], y[cutoff:]
        test_df = df.iloc[cutoff:].copy()

        rf = train_rf_model(X_train, y_train)
        y_pred = rf.predict(X_test)

        # Save predictions to test DataFrame.
        test_df = test_df.assign(Predicted_Return=y_pred)
        # Use a naive baseline of 0 return (i.e., no change)
        test_df = test_df.assign(Baseline_Return=np.zeros(len(test_df)))

        metrics_model = evaluate_model(test_df["Target_Return"].values, test_df["Predicted_Return"].values)
        metrics_baseline = evaluate_model(test_df["Target_Return"].values, test_df["Baseline_Return"].values)

        mse_improv = (1 - (metrics_model["mse"] / metrics_baseline["mse"])) * 100 if metrics_baseline["mse"] != 0 else 0
        mae_improv = (1 - (metrics_model["mae"] / metrics_baseline["mae"])) * 100 if metrics_baseline["mae"] != 0 else 0

        # Reconstruct the predicted price using the predicted cumulative return.
        test_df = reconstruct_prices(test_df, horizon)
        price_metrics = evaluate_model(test_df["Actual_Future_Price"].values, test_df["Predicted_Price"].values)

        st.subheader("Performance (Cumulative Returns)")
        st.write(f"RF MSE: {metrics_model['mse']:.4f}")
        st.write(f"RF MAE: {metrics_model['mae']:.4f}")
        st.write(f"Baseline MSE (0 return): {metrics_baseline['mse']:.4f}")
        st.write(f"Baseline MAE (0 return): {metrics_baseline['mae']:.4f}")
        st.write(f"MSE Improvement: {mse_improv:.2f}%")
        st.write(f"MAE Improvement: {mae_improv:.2f}%")

        st.subheader("Performance (Price Forecast)")
        st.write(f"Price MSE: {price_metrics['mse']:.2f}")
        st.write(f"Price MAE: {price_metrics['mae']:.2f}")

        st.subheader("Predicted vs. Actual Future Prices (Last 30 days of Test)")
        last_df = test_df.tail(30)
        fig, ax = plt.subplots()
        ax.plot(last_df.index, last_df["Actual_Future_Price"], label="Actual Future Price")
        ax.plot(last_df.index, last_df["Predicted_Price"], label="Predicted Price")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
