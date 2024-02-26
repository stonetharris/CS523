import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def generate_data(n_samples=300):
    # blob of 2D points
    data, _ = make_blobs(n_samples=n_samples, centers=3, n_features=2, cluster_std=0.5, random_state=42)
    data = (data - data.min()) / (data.max() - data.min())
    return data

data = generate_data()

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(2, 4), 
            nn.ReLU(),
            nn.Linear(4, 2)  
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),  
            nn.ReLU(),
            nn.Linear(4, 2)   
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

# Convert the data to PyTorch tensors
tensor_data = torch.Tensor(data)

# create dataset
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# initialize + opt
autoencoder = Autoencoder()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

# training
num_epochs = 100
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, = batch
        optimizer.zero_grad()
        
        # forward + backward pass then optimize
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
    
    # Print statistics
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

autoencoder.eval()
with torch.no_grad():
    reconstructed = autoencoder(tensor_data).numpy()

plt.figure(figsize=(12, 6))

# original data
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.title('Original 2D Dataset')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# reconstructed data
plt.subplot(1, 2, 2)
plt.scatter(reconstructed[:, 0], reconstructed[:, 1], alpha=0.5)
plt.title('Reconstructed 2D Dataset')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

plt.show()