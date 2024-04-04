import alpaca_trade_api as tradeapi
from requests.exceptions import HTTPError

# Replace 'your_api_key' and 'your_api_secret' with your actual Alpaca API credentials
ALPACA_API_KEY = 'CK856BN16OIJ1ABNAOPH'
ALPACA_API_SECRET = '1VjHK2vTS9NsN89t1mJafoD2acXRhRLFYacPkhnx'
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # Use the paper trading URL

# Initialize the API connection
alpaca_api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, base_url=ALPACA_BASE_URL)

# Fetch historical data for the S&P 500 index (using the symbol SPY for this example)
symbol = 'AAPL'  # You can replace this with any ticker
timeframe = '1Day'  # Use '1Day' for daily data; you can also use '1Min', '5Min', '15Min', etc.

# Get historical data
# The get_bars call is paginated, you can fetch more data by calling it several times
# historical_data = alpaca_api.get_bars(symbol, timeframe).df

try:
    historical_data = alpaca_api.get_bars(symbol, timeframe).df
    for index, row in historical_data.iterrows():
        print(f"Date: {index}, Open: {row['open']}, High: {row['high']}, Low: {row['low']}, Close: {row['close']}, Volume: {row['volume']}")
except HTTPError as http_err:
    # If a HTTP error occurs, print the details
    print(f"HTTP error occurred: {http_err}")  # Python 3.6+
    print(f"Response body: {http_err.response.text}")  # Print the text content of the response

# Print the data
# for index, row in historical_data.iterrows():
#     print(f"Date: {index}, Open: {row['open']}, High: {row['high']}, Low: {row['low']}, Close: {row['close']}, Volume: {row['volume']}")