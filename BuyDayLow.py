from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from datetime import datetime, timedelta
import pandas as pd
import pandas_ta as ta
import numpy as np


class RangeStrategy(Strategy):
    def initialize(self, pct_thresh=50):
        self.pct_thresh = pct_thresh

    def on_trading_iteration(self):
        symbol = "SPY"  # Example symbol, replace with your desired symbol
        data = self.get_historical_prices(symbol, 2)  # Get 2 days of data

        if len(data) < 2:
            return

        df = pd.DataFrame(data)

        # Calculate additional columns for strategy
        df['Range'] = df['high'] - df['low']
        df['Dist'] = abs(df['close'] - df['low'])
        df['Pct'] = (df['Dist'] / df['Range']) * 100

        # Identify entries
        long_condition = (df['Pct'].iloc[-1] < self.pct_thresh) and (df['Range'].iloc[-1] > 10)

        if long_condition and not self.get_position(symbol):
            qty = self.portfolio_value // df['close'].iloc[-1]
            self.submit_order(symbol, qty, "buy")
        elif not long_condition and self.get_position(symbol):
            self.submit_order(symbol, self.get_position(symbol).quantity, "sell")


def backtest(symbol, start_date, end_date, starting_balance, pct_thresh):
    strategy = RangeStrategy(
        name='RangeStrategy',
        budget=starting_balance,
        parameters={"pct_thresh": pct_thresh}
    )

    backtesting_engine = YahooDataBacktesting(
        strategy,
        start_date,
        end_date,
        benchmark_asset=symbol
    )

    results = backtesting_engine.run()
    return results


# Set your parameters
START = datetime(2020, 1, 1)
END = datetime(2023, 1, 1)
STARTING_BALANCE = 10000
PCT_THRESH = 50
SYMBOL = "SPY"

# Run backtest
results = backtest(SYMBOL, START, END, STARTING_BALANCE, PCT_THRESH)

# Extract metrics
bench_cagr = results.benchmark_cagr
sys_cagr = results.cagr
bench_dd = results.benchmark_max_drawdown
sys_dd = results.max_drawdown

print(f"Benchmark CAGR: {bench_cagr:.2f}%")
print(f"System CAGR: {sys_cagr:.2f}%")
print(f"Benchmark Max Drawdown: {bench_dd:.2f}%")
print(f"System Max Drawdown: {sys_dd:.2f}%")