from ib_insync import *

# Initialize the IB class
ib = IB()

# Connect to the Interactive Brokers TWS API
# The default IP is '127.0.0.1', port is 7497 for paper trading, clientId can be any unique integer
ib.connect('127.0.0.1', 7497, clientId=1)

# Test connection by fetching current market data for the S&P 500 futures (symbol ES)
contract = Future('ES', '202406', 'CME')
ib.qualifyContracts(contract)

apple_contract = Stock('AAPL', 'NASDAQ', 'USD')

# Fetch historical data for the contract
historical_data = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='1 W',  # 1 week of historical data
    barSizeSetting='1 day',  # 1 day intervals
    whatToShow='MIDPOINT',
    useRTH=True
)

# Convert to a pandas DataFrame and print it
df = util.df(historical_data)
print(df)

# Disconnect from the API
ib.disconnect()
