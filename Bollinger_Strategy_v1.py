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
QUANTITY = 1000
START_DATE = "2024-08-19"
END_DATE = "2024-07-19"
SLEEP_TIME="60M"
TIMESTEP="60 minutes"

#TIMESTEP="day"
#SLEEP_TIME="1D"

INIT_SIZE=0.50



RSI_TOP_LIMIT=50
RSI_BOTTOM_LIMIT=50
HOLDING_PERIOD=5
def addemasignal(df, backcandles):
    emasignal = [0]*len(df)
    for row in range(backcandles, len(df)):
        upt = 1
        dnt = 1
        for i in range(row-backcandles, row+1):
            if df.high.iloc[i] >= df.EMA.iloc[i]:
                dnt = 0
            if df.low.iloc[i] <= df.EMA.iloc[i]:
                upt = 0
        if upt == 1 and dnt == 1:
            emasignal[row] = 3
        elif upt == 1:
            emasignal[row] = 2
        elif dnt == 1:
            emasignal[row] = 1
    df['EMASignal'] = emasignal

def addorderslimit(df, percent):
    ordersignal = [0] * len(df)
    for i in range(1, len(df)):  # EMASignal of previous candle!!! modified!!!
        if df.EMASignal.iloc[i] == 2 and df.close.iloc[i] <= df['BBL_20_2.5'].iloc[i]:
            ordersignal[i] = df.close.iloc[i] - df.close.iloc[i] * percent
        elif df.EMASignal.iloc[i] == 1 and df.close.iloc[i] >= df['BBU_20_2.5'].iloc[i]:  # and df.RSI[i]>=0:
            ordersignal[i] = df.close.iloc[i] + df.close.iloc[i] * percent
    df['ordersignal'] = ordersignal

class Trend(Strategy):

    initsize = INIT_SIZE
    ordertime = []
    holding_period = HOLDING_PERIOD

    def initialize(self):
        start = START_DATE
        self.signal = 0
        self.start = start
        self.sleeptime = SLEEP_TIME
        self.symbol = SYMBOL
        self.last_trade_time = None

    # minute bars, make functions

    def on_trading_iteration(self, elseif=None):
        bars = self.get_historical_prices(self.symbol, 200, timestep="15 minutes")
        data = bars.df
        close_data = data["close"]

        # Calculate indicators
        data['EMA'] = ta.sma(close_data, length=200)
        data['RSI'] = ta.rsi(close_data, length=2)
        my_bbands = ta.bbands(close_data, length=20, std=2.5)
        data = data.join(my_bbands)
        #data.dropna(inplace=True)

        # Add EMA Signal
        addemasignal(data, 6)

        # Add Order Limits based on Bollinger Bands
        addorderslimit(data, 0.03)  # Adjust the percent if necessary

        # Update the latest signal
        self.signal = data['ordersignal'].iloc[-1]
        self.ema = data['EMASignal'].iloc[-1]


        # Exit conditions for current trade
        pos = self.get_position(self.symbol)
        if pos is not None:
            current_order = pos.orders[0]
            current_rsi = data['RSI'].iloc[-1]
            if self.ordertime:
                holding_duration = (self.get_datetime() - self.ordertime[0]).days
                if holding_duration >= self.holding_period or \
                        (current_order.side == "buy" and current_rsi >= RSI_TOP_LIMIT) or \
                        (current_order.side == "sell" and current_rsi <= RSI_BOTTOM_LIMIT):
                    self.sell_all()
                    if self.ordertime:
                        self.ordertime.pop(0)


        # Entry logic based on signal
        if self.signal != 0 and pos is None:
            self.sell_all()
            if self.ema == 2:
                print("---Buy----", self.get_datetime(), self.signal)
                order = self.create_order(self.symbol, QUANTITY, "buy")
                self.submit_order(order)
                self.ordertime.append(self.get_datetime())
            elif self.ema == 1:
                print("---Sell---", self.get_datetime(), self.signal)
                order = self.create_order(self.symbol, QUANTITY, "sell")
                self.submit_order(order)
                self.ordertime.append(self.get_datetime())





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