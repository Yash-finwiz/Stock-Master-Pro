import numpy as np
from sklearn import preprocessing, model_selection
import yfinance as yf
import datetime as dt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def predict_lstm(ticker_value, number_of_days):
    # Extract features and target variable
    df_ml = yf.download(tickers=ticker_value, period='3mo', interval='1h')
    df_ml = df_ml[['Adj Close']]
    forecast_out = int(number_of_days)
    df_ml['Prediction'] = df_ml[['Adj Close']].shift(-forecast_out)

        # Split data into training and testing sets
    X = np.array(df_ml.drop(['Prediction'], 1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df_ml['Prediction'])
    y = y[:-forecast_out]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
        # Define the LSTM model architecture
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

        # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

        # Reshape the input data for LSTM model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Fit the model to the training data
    model.fit(X_train, y_train, epochs=10, batch_size=32)

        # Reshape the input data for evaluation
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Evaluate the model on the testing data
        #loss = model.evaluate(X_test, y_test)

        # Make predictions on the forecast data
    X_forecast = np.reshape(X_forecast, (X_forecast.shape[0], X_forecast.shape[1], 1))
    forecast_prediction = model.predict(X_forecast)
    forecast = forecast_prediction.flatten().tolist()
    return forecast