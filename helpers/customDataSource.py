from lumibot.backtesting import YahooDataBacktesting

class ModifiedYahooDataBacktesting(YahooDataBacktesting):
    def __init__(self,benchmark, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark_data = None
        self.BENCHMARK = benchmark
    def get_historical_prices(self, symbol, start_date, end_date, timeframe="DAY"):
        if symbol == self.BENCHMARK:
            return self.benchmark_data
        return super().get_historical_prices(symbol, start_date, end_date, timeframe)
