import numpy as np
import pandas as pd
from ib_insync import *

# Initialize the IB class and connect as before
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define the stock and fetch historical data
stock = Stock('AAPL', 'SMART', 'USD')
bars = ib.reqHistoricalData(stock, endDateTime='', durationStr='2 M', barSizeSetting='1 day', whatToShow='MIDPOINT', useRTH=True)

# Convert to a pandas DataFrame
df = util.df(bars)
df['Date'] = pd.to_datetime(df['date'])

# Calculate moving averages
short_window = 20
long_window = 50
df['Short_Moving_Avg'] = df['close'].rolling(window=short_window, min_periods=1).mean()
df['Long_Moving_Avg'] = df['close'].rolling(window=long_window, min_periods=1).mean()

# Generate signals
df['Signal'] = 0.0
df['Signal'][short_window:] = np.where(df['Short_Moving_Avg'][short_window:] > df['Long_Moving_Avg'][short_window:], 1.0, 0.0)
df['Position'] = df['Signal'].diff()

print(df[['Date', 'close', 'Short_Moving_Avg', 'Long_Moving_Avg', 'Signal', 'Position']])

# Disconnect from the API
ib.disconnect()
