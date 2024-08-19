from helpers.config import ALPACA_CONFIG
from datetime import datetime, timedelta
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.strategies import Strategy
from lumibot.traders import Trader
import numpy as np
import pandas_ta as ta
from indicators.RSI import calculate_rsi
from indicators.MovingAverage import calculate_avg
import pandas as pd



SYMBOL = "USDJPY=X"
QUANTITY = 100
START_DATE = "2024-05-01"
END_DATE = "2024-05-30"
#SLEEP_TIME="240M"
SLEEP_TIME="1D"

class Trend(Strategy):

    def initialize(self):
        signal = None
        self.signal = signal
        self.start = start
        self.sleeptime = SLEEP_TIME
        self.symbol = SYMBOL

        self.length = 22
        self.mult = 3.0
        self.use_close = True
        self.dir = 1

    # minute bars, make functions



    def on_trading_iteration(self):
        bars = self.get_historical_prices(self.symbol, 50 + self.length)
        data = bars.df

        # Calculate ATR
        data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=self.length) * self.mult

        # Calculate long and short stops
        if self.use_close:
            data['highest'] = data['close'].rolling(window=self.length).max()
            data['lowest'] = data['close'].rolling(window=self.length).min()
        else:
            data['highest'] = data['high'].rolling(window=self.length).max()
            data['lowest'] = data['low'].rolling(window=self.length).min()

        data['long_stop'] = data['highest'] - data['atr']
        data['short_stop'] = data['lowest'] + data['atr']

        data['updated_long_stop'] = data['long_stop']
        data['updated_short_stop'] = data['short_stop']

        for i in range(1, len(data)):
            if data['close'].iloc[i - 1] > data['updated_long_stop'].iloc[i - 1]:
                data.loc[data.index[i], 'updated_long_stop'] = max(data['long_stop'].iloc[i],
                                                                   data['updated_long_stop'].iloc[i - 1])
            else:
                data.loc[data.index[i], 'updated_long_stop'] = data['long_stop'].iloc[i]

            if data['close'].iloc[i - 1] < data['updated_short_stop'].iloc[i - 1]:
                data.loc[data.index[i], 'updated_short_stop'] = min(data['short_stop'].iloc[i],
                                                                    data['updated_short_stop'].iloc[i - 1])
            else:
                data.loc[data.index[i], 'updated_short_stop'] = data['short_stop'].iloc[i]

        data['dir'] = np.where(data['close'] > data['updated_short_stop'].shift(1), 1,
                               np.where(data['close'] < data['updated_long_stop'].shift(1), -1, np.nan))
        data['dir'] = data['dir'].fillna(method='ffill')

        # Check for buy and sell signals
        data['buy_signal'] = (data['dir'] == 1) & (data['dir'].shift(1) == -1)
        data['sell_signal'] = (data['dir'] == -1) & (data['dir'].shift(1) == 1)

        current_dir = data['dir'].iloc[-1]
        buy_signal = data['buy_signal'].iloc[-1]
        sell_signal = data['sell_signal'].iloc[-1]

        if buy_signal:
            self.sell_all()  # Close any existing short positions
            order = self.create_order(self.symbol, 100, "buy")  # Adjust quantity as needed
            self.submit_order(order)

        elif sell_signal:
            self.sell_all()  # Close any existing long positions
            order = self.create_order(self.symbol, 100, "sell")  # Adjust quantity as needed
            self.submit_order(order)

            # Update the strategy's direction
        self.dir = current_dir


        self.log_message(f"Direction: {self.dir}, Buy Signal: {buy_signal}, Sell Signal: {sell_signal}")



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
            end,
            parameters={"symbol": SYMBOL},
        )