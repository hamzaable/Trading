import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies import Strategy
from lumibot.traders import Trader
from lumibot.backtesting import BacktestingBroker
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from helpers.FetchData import fetch_historical_data
from helpers.customDataSource import ModifiedYahooDataBacktesting

# Configuration Parameters
SYMBOL = 'USDJPY=X'
START_DATE = "2021-01-01"
END_DATE = "2024-06-01"
BENCHMARK = 'SPY'

# Fetch historical data from Yahoo Finance
data = fetch_historical_data([SYMBOL, BENCHMARK], START_DATE, END_DATE, '1d')

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

class RSIMACDStrategy(Strategy):
    def initialize(self):
        self.symbol = SYMBOL
        self.sleeptime = '1d'
        self.cash_at_risk = 0.5
        self.last_trade = "NONE"

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        if last_price is None or last_price <= 0:
            return cash, last_price, 0
        quantity = int((cash * self.cash_at_risk) / last_price)
        return cash, last_price, max(0, quantity)

    def on_trading_iteration(self):
        current_price = float(self.get_last_price(self.symbol))
        if current_price is None:
            self.log_message(f"Unable to get current price for {self.symbol}")
            return

        # Fetch historical data
        historical_data = data[SYMBOL]
        if historical_data is None or historical_data.empty:
            self.log_message(f"Unable to get historical data for {self.symbol}")
            return

        # Calculate RSI and MACD
        historical_data['RSI'] = calculate_rsi(historical_data)
        historical_data['MACD'], historical_data['Signal_Line'] = calculate_macd(historical_data)

        current_rsi = historical_data['RSI'].iloc[-1]
        current_macd = historical_data['MACD'].iloc[-1]
        current_signal = historical_data['Signal_Line'].iloc[-1]
        prev_macd = historical_data['MACD'].iloc[-2]
        prev_signal = historical_data['Signal_Line'].iloc[-2]

        cash, last_price, quantity = self.position_sizing()

        if quantity == 0:
            self.log_message("Quantity is 0, skipping trade")
            return

        # Execute trades based on RSI and MACD
        if current_rsi > 80 and current_macd > current_signal and prev_macd < prev_signal:
            if self.get_position(self.symbol):
                self.sell_all()
                self.last_trade = "sell"
        elif current_rsi < 30 and current_macd < current_signal and prev_macd > prev_signal:
            if not self.get_position(self.symbol):
                order = self.create_order(
                    self.symbol,
                    quantity,
                    side="buy",
                    type="bracket",
                    take_profit_price=last_price * 1.20,
                    stop_loss_price=last_price * 0.95
                )
                self.submit_order(order)
                self.last_trade = "buy"

start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
end_date = datetime.strptime(END_DATE, "%Y-%m-%d")

data_source = ModifiedYahooDataBacktesting(
    BENCHMARK,
    start_date,
    end_date,
)

data_source.benchmark_data = data[BENCHMARK]

broker = BacktestingBroker(data_source)

strategy = RSIMACDStrategy(name='RSI MACD Strategy', broker=broker,
                           parameters={"symbol": SYMBOL,
                                       "cash_at_risk": 0.5})

strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": SYMBOL},
    show_plot=True,
    show_tearsheet=True,
    benchmark_asset=BENCHMARK
)