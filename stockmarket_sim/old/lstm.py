import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ib_insync import *

# Connect to Interactive Brokers API
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Fetch historical data
stock = Stock('AAPL', 'SMART', 'USD')
bars = ib.reqHistoricalData(
    stock,
    endDateTime='',
    durationStr='1 Y',  # 1 year of historical data
    barSizeSetting='1 day',  # 1 day intervals
    whatToShow='MIDPOINT',
    useRTH=True
)

# Disconnect after fetching data
ib.disconnect()

# Convert to a pandas DataFrame
df = util.df(bars)

# Extract the closing prices
data = df['close'].values.astype(float).reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Prepare data for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(scaled_data, seq_length)
X_train = torch.from_numpy(X[:int(len(X)*0.8)]).type(torch.Tensor)
X_test = torch.from_numpy(X[int(len(X)*0.8):]).type(torch.Tensor)
y_train = torch.from_numpy(y[:int(len(y)*0.8)]).type(torch.Tensor)
y_test = torch.from_numpy(y[int(len(y)*0.8):]).type(torch.Tensor)

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10

for i in range(epochs):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%2 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

# Predicting
model.eval()
with torch.no_grad():
    preds = []
    for seq in X_test:
        preds.append(model(seq).item())

# Inverse transform the predictions
actual_predictions = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

# Now you can compare 'actual_predictions' with the actual prices, plot them, or use them in your trading strategy
