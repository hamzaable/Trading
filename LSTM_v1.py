import keras.src.optimizers
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler


import tensorflow as tf
import keras
from keras import layers

from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies import Strategy


# Data retrieval and preprocessing
def get_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    return df


# Feature engineering
def add_technical_indicators(df):
    df['RSI'] = ta.rsi(df['Close'])
    df['MACD'] = ta.macd(df['Close'])['MACD']
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.bbands(df['Close'])
    df.dropna(inplace=True)
    return df


# LSTM model creation
def create_lstm_model(input_shape):
    model = keras.Sequential([
        layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(50, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.src.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


# Prepare data for LSTM
def prepare_data(df, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    return X, y, scaler


# Signal filtering
def filter_signals(predictions, actual, threshold=0.02):
    signals = np.zeros_like(predictions)
    signals[predictions > actual * (1 + threshold)] = 1  # Buy signal
    signals[predictions < actual * (1 - threshold)] = -1  # Sell signal
    return signals


# LSTM Strategy
class LSTMStrategy(Strategy):
    def initialize(self, symbol, model, scaler, look_back):
        self.symbol = symbol
        self.model = model
        self.scaler = scaler
        self.look_back = look_back
        self.last_signal = 0

    def on_trading_iteration(self):
        current_price = self.get_last_price(self.symbol)

        # Get historical data
        historical_data = self.get_historical_data(self.symbol, 100)  # Adjust the number of candles as needed
        df = add_technical_indicators(historical_data)

        # Prepare data for prediction
        features = df[['Close', 'RSI', 'MACD', 'ATR', 'BB_upper', 'BB_middle', 'BB_lower']].values
        scaled_features = self.scaler.transform(features)
        X = np.array([scaled_features[-self.look_back:]])

        # Make prediction
        prediction = self.model.predict(X)
        unscaled_prediction = self.scaler.inverse_transform(prediction)[0, 0]

        # Generate and filter signal
        signal = filter_signals(unscaled_prediction, current_price, threshold=0.02)

        if signal == 1 and self.last_signal != 1:
            self.buy(self.symbol)
            self.last_signal = 1
        elif signal == -1 and self.last_signal != -1:
            self.sell(self.symbol)
            self.last_signal = -1


backtesting_start = "2023-01-01"
backtesting_end = "2023-12-31"

# Main execution
if __name__ == "__main__":
    symbol = "AAPL"
    start_date = backtesting_start
    end_date = backtesting_end

    # Get and preprocess data
    df = get_data(symbol, start_date, end_date)
    df = add_technical_indicators(df)

    # Prepare data for LSTM
    features = df[['Close', 'RSI', 'MACD', 'ATR', 'BB_upper', 'BB_middle', 'BB_lower']]
    X, y, scaler = prepare_data(features)

    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Create and train LSTM model
    model = create_lstm_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

    # Backtesting
    strategy = LSTMStrategy(
        symbol=symbol,
        model=model,
        scaler=scaler,
        look_back=60
    )



    backtest = YahooDataBacktesting(
        strategy,
        backtesting_start,
        backtesting_end,
        benchmark_asset="SPY"
    )

    results = backtest.run()
    backtest.plot_results()