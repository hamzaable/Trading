import yfinance as yf

def fetch_historical_data(symbols, start_date, end_date, interval):
    data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        symbol_data = ticker.history(start=start_date, end=end_date, interval=interval)
        if symbol_data.empty:
            raise ValueError(f"No historical data found for {symbol} in the specified date range.")
        data[symbol] = symbol_data
    return data