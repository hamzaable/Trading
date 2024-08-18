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

# Configuration Parameters
WINDOW_SIZE = 30
INTERVAL = '1d'
SYMBOL = 'AAPL'
START_DATE = "2024-01-01"
END_DATE = "2024-06-01"
BENCHMARK = 'SPY'
# Fetch historical data from Yahoo Finance
def fetch_historical_data(symbols, start_date, end_date, interval):
    data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        symbol_data = ticker.history(start=start_date, end=end_date, interval=interval)
        if symbol_data.empty:
            raise ValueError(f"No historical data found for {symbol} in the specified date range.")
        data[symbol] = symbol_data
    return data

data = fetch_historical_data([SYMBOL, BENCHMARK], START_DATE, END_DATE, INTERVAL)


class ModifiedYahooDataBacktesting(YahooDataBacktesting):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark_data = None

    def get_historical_prices(self, symbol, start_date, end_date, timeframe="DAY"):
        if symbol == BENCHMARK:
            return self.benchmark_data
        return super().get_historical_prices(symbol, start_date, end_date, timeframe)



# Identify order blocks
def identify_order_blocks(data: pd.DataFrame, window_size: int):
    order_blocks = []

    for i in range(window_size, len(data) - 1):
        current_window = data['Close'].iloc[i - window_size:i]
        current_close = data['Close'].iloc[i]

        if not current_window.empty:
            high = current_window.max()
            low = current_window.min()
            if current_close > high or current_close < low:
                order_blocks.append((data.index[i], current_close))

    return order_blocks

# Analyze market structure
def analyze_market_structure(data, window_size):
    trends = []
    for i in range(window_size, len(data) - 1):
        current_window = data['Close'].iloc[i - window_size:i]
        current_close = data['Close'].iloc[i]

        if len(current_window) > 1:
            last_window_price = current_window.iloc[-1]
            if current_close > last_window_price:
                trends.append((last_window_price, 'Uptrend'))
            else:
                trends.append((last_window_price, 'Downtrend'))
    return trends

# Make trading decision based on BTMM strategy
def make_trading_decision(order_blocks, trends, current_price):
    decision = "hold"
    current_price = Decimal(str(current_price))

    # Example order block-based decision
    for block in order_blocks:
        try:
            block_price = Decimal(str(block[1]))
            if current_price < block_price * Decimal("0.99"):
                decision = "buy"
            elif current_price > block_price * Decimal("1.01"):
                decision = "sell"
        except (InvalidOperation, ValueError, TypeError):
            print(f"Error processing block price: {block[1]}")
            continue

    # Example trend-based decision
    if trends:
        last_trend = trends[-1]
        try:
            last_trend_price = Decimal(str(last_trend[0]))
            if last_trend[1] == 'Uptrend' and current_price > last_trend_price * Decimal("1.01"):
                decision = "buy"
            elif last_trend[1] == 'Downtrend' and current_price < last_trend_price * Decimal("0.99"):
                decision = "sell"
        except (InvalidOperation, ValueError, TypeError):
            print(f"Error processing trend price: {last_trend[0]}")
            pass

    return decision

class BTMMStrategy(Strategy):
    def initialize(self):
        self.symbol = SYMBOL
        self.window_size = WINDOW_SIZE
        self.sleeptime = INTERVAL
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

        # Identify order blocks and analyze market structure
        order_blocks = identify_order_blocks(historical_data, self.window_size)
        trends = analyze_market_structure(historical_data, self.window_size)

        # Make trading decision
        decision = make_trading_decision(order_blocks, trends, current_price)
        cash, last_price, quantity = self.position_sizing()

        if quantity == 0:
            self.log_message("Quantity is 0, skipping trade")
            return

        # Execute trades based on the decision
        if decision == "buy" and not self.get_position(self.symbol):
            if self.last_trade == "sell":
                self.sell_all()
            order = self.create_order(
                self.symbol,
                quantity,
                "buy",
                type="bracket",
                take_profit_price=last_price * 1.20,
                stop_loss_price=last_price * 0.95
            )
            self.submit_order(order)
            self.last_trade = "buy"
        elif decision == "sell" and self.get_position(self.symbol):
            if self.last_trade == "buy":
                self.sell_all()
            order = self.create_order(
                self.symbol,
                quantity,
                "sell",
                type="bracket",
                take_profit_price=last_price * 0.80,
                stop_loss_price=last_price * 1.05
            )
            self.submit_order(order)
            self.last_trade = "sell"


start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
end_date = datetime.strptime(END_DATE, "%Y-%m-%d")

data_source = ModifiedYahooDataBacktesting(
    start_date,
    end_date
)

data_source.benchmark_data = data[BENCHMARK]

broker = BacktestingBroker(data_source)

strategy = BTMMStrategy(name='BTMM Strategy', broker=broker,
                        parameters={"symbol": SYMBOL,
                                    "cash_at_risk": 0.5})

strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": SYMBOL},
    show_plot=True,
    show_tearsheet=True,
    benchmark_asset="SPY"
)
