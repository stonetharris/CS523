from ib_insync import *
import datetime, time

time = datetime.datetime.now()

# Connect to Interactive Brokers API
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

contract = Future('ES', '202406', 'CME')
ib.qualifyContracts(contract)

ib.reqMarketDataType(3)

historical_data = ib.reqHistoricalData(
    contract, endDateTime='', durationStr='30 D',
    barSizeSetting='1 hour', whatToShow='MIDPOINT', useRTH=True)

df = util.df(historical_data)
# print(df)
df.to_csv(f"data/{time}.csv", index =False)

data = df[['close']].values.astype(float)

# Disconnect from the API
ib.disconnect()

