from helpers.config import ALPACA_CONFIG
from datetime import datetime, timedelta
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.strategies import Strategy
from lumibot.traders import Trader
import numpy as np
from indicators.RSI import calculate_rsi
from indicators.MovingAverage import calculate_avg
import pandas_ta as ta
import pandas as pd



SYMBOL = "SPY"
QUANTITY = 100
START_DATE = "2020-01-01"
END_DATE = "2024-06-01"
SLEEP_TIME="240M"

class Trend(Strategy):

    def initialize(self):
        signal = None
        start = START_DATE
        self.signal = signal
        self.start = start
        self.sleeptime = SLEEP_TIME
        self.symbol = SYMBOL

    # minute bars, make functions



    def on_trading_iteration(self):
        bars = self.get_historical_prices(self.symbol, 50)
        data = bars.df
        close_data = data["close"]
        # gld = pd.DataFrame(yf.download("GLD", self.start)['Close'])
        data["RSI"] =  ta.rsi(close_data, length=16)
        current_rsi = data['RSI'].iloc[-1]
        data['5-day'] = ta.sma(close_data, 5).iloc[-1]
        data['12-day'] = ta.sma(close_data, 12).iloc[-1]

        data['Signal'] = np.where(np.logical_and(current_rsi > 70, data['5-day'] > data['12-day']),"BUY", None)
        data['Signal'] = np.where(np.logical_and(current_rsi < 30, data['5-day'] < data['12-day']),"SELL", data['Signal'])

        self.signal = data.iloc[-1].Signal

        symbol = self.symbol
        if self.signal == 'BUY':
            pos = self.get_position(symbol)
            if pos is not None:
                self.sell_all()
            order = self.create_order(symbol, QUANTITY, "buy")
            self.submit_order(order)

        elif self.signal == 'SELL':
            pos = self.get_position(symbol)
            if pos is not None:
                self.sell_all()
            order = self.create_order(symbol, QUANTITY, "sell")
            self.submit_order(order)


if __name__ == "__main__":
    trade = False
    if trade:
        broker = Alpaca(ALPACA_CONFIG(True))
        strategy = Trend(broker=broker)
        bot = Trader()
        bot.add_strategy(strategy)
        bot.run_all()
    else:
        start = datetime.strptime(START_DATE, "%Y-%m-%d")
        end = datetime.strptime(END_DATE, "%Y-%m-%d")
        Trend.backtest(
            YahooDataBacktesting,
            start,
            end
        )