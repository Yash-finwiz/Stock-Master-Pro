import numpy as np
import pandas as pd

from sklearn import preprocessing, model_selection
import yfinance as yf
import datetime as dt
from statsmodels.tsa.arima.model import ARIMA





def predict_arima(ticker_value, number_of_days):
    df_arima = yf.download(tickers=ticker_value, period='1y', interval='1d')
    df_arima = df_arima['Adj Close']

        # Fit ARIMA model
    order = (5, 1, 0)  # Example order, you may need to fine-tune this
    model = ARIMA(df_arima, order=order)
    results = model.fit()

        # Forecast future values
    forecast_model = results.get_forecast(steps=number_of_days)
    forecast_index = pd.date_range(df_arima.index[-1], periods=number_of_days + 1, freq='B')[1:]
    forecast = forecast_model.predicted_mean.values
    return forecast