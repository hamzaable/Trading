def calculate_avg(data, length=9):
   return data['close'].rolling(length).mean()