import pandas as pd


class CustomDataSource(BacktestingDataSource):
    def __init__(self, start_date, end_date, data_file):
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.read_csv(data_file, parse_dates=['Date'])
        self.data = self.data[(self.data['Date'] >= self.start_date) & (self.data['Date'] <= self.end_date)]

    def get_historical_prices(self, symbol, start=None, end=None, interval='1d'):
        if start is None:
            start = self.start_date
        if end is None:
            end = self.end_date

        # Filter data for the symbol and date range
        data_filtered = self.data[(self.data['Symbol'] == symbol) &
                                  (self.data['Date'] >= start) &
                                  (self.data['Date'] <= end)]

        return data_filtered.set_index('Date').resample(interval).last()

    def get_last_price(self, symbol):
        # Get the last price for the symbol from the data
        last_row = self.data[self.data['Symbol'] == symbol].iloc[-1]
        return last_row['Close']

    def get_chains(self, symbol):
        # Placeholder implementation, since options chains might not be relevant
        return None
