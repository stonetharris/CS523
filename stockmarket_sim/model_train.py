import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import json
import datetime
import pandas as pd

# hyperparemeters ---- tune these
seq_length = 10
hidden_dim = 32  # try 8,16,32,64
num_layers = 2
lr = 0.01  # try 3e-4
num_epochs = 1_000  # try 1_000
interval = "1h"
train_length = f"{.8*30} days"

# variables
input_dim = 1
output_dim = 1

filename = "2024-04-03 14:09:36.731368.csv"
df = pd.read_csv(f"data/{filename}")

scaler = MinMaxScaler()
data = df[["close"]].values.astype(float)
data_normalized = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


X, y = create_sequences(data_normalized, seq_length)
X = torch.from_numpy(X).type(torch.Tensor)
y = torch.from_numpy(y).type(torch.Tensor)

train_size = int(len(X) * 0.8)
test_size = len(X) - train_size

train_X, test_X = X[:train_size], X[train_size:]
train_y, test_y = y[:train_size], y[train_size:]

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out[:, -1, :]) 
        return out

torch.manual_seed(42)

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)

loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train the model
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(train_X)
    # Compute Loss
    loss = loss_fn(y_pred, train_y)

    # Backward pass
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item()}")
        model.eval()

        val_pred = model(test_X)
        # convert to numpy
        val_pred = val_pred.detach().numpy()
        val_pred = scaler.inverse_transform(val_pred)
        val_y = scaler.inverse_transform(test_y)
        val_loss = np.mean(np.abs(val_pred - val_y))
        print(f"Validation Loss: {val_loss.item()}")

# Evaluate the model
model.eval()
test_pred = model(test_X)
test_loss = loss_fn(test_pred, test_y)
training_loss = loss_fn(
    model(train_X), train_y
) #this may cause error, want to track training loss
print(f'Test Loss: {training_loss.item()}') #this line goes with the preceeding one^
print(f'Test Loss: {test_loss.item()}')

# Predicting the next hour's close price
last_sequence = data_normalized[-seq_length:].reshape((1, seq_length, 1))
last_sequence = torch.from_numpy(last_sequence).type(torch.Tensor)
with torch.no_grad():
    predicted_normalized = model(last_sequence)
predicted = scaler.inverse_transform(predicted_normalized.numpy())

print(f"Predicted next hour's close price: {predicted.flatten()[0]}")

def generate(last_sequence):
    """
    returns predicted price ,
    """

    last_sequence_torch = torch.from_numpy(last_sequence).type(torch.Tensor)

    with torch.no_grad():
        predicted_normalized = model(last_sequence_torch)
    # use predicted price as if it were the true price for future prediction
    output_array = list(last_sequence[0, :, 0]) + [predicted_normalized.numpy().item()]

    output_array = np.array(output_array[1:]).reshape(1, seq_length, 1)
    z = scaler.inverse_transform(predicted_normalized.numpy())
    return z.item(), output_array


x = data_normalized[-seq_length:].reshape((1, seq_length, 1))
prices = []
for ind in range(10):
    price, x = generate(x)

    prices.append(round(price, 2))
    # print(price)

# use json to save into a metadata folder
metadata = {
    "seq_length": seq_length,
    "hidden_dim": hidden_dim,
    "num_layers": num_layers,
    "lr": lr,
    "num_epochs": num_epochs,
    "training_loss": round(training_loss.item(), 2),
    "test_loss": round(test_loss.item(), 2),
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "prices": prices,
    "interval": interval,
    "train length": train_length,
    "Average dollar ammount off on validation": round(val_loss.item(), 2),
}

# if train loss is much lower than test loss, likely overfitting
filename = f"metadata/metadata.json"
# append current metadata to the file
with open(filename, "r") as f:
    current = json.load(f)
    current.append(metadata)
with open(filename, "w") as f:
    json.dump(current, f)