import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from gameEnv import MyTetrisEnv  # Custom Tetris environment


# Define the neural network model
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return x


# Hyperparameters
input_size = 200  # Example input size (state space dimension)
output_size = 1  # Example output size (number of actions)
learning_rate = 0.001
num_epochs = 1000
batch_size = 32

# Initialize Tetris environment
env = MyTetrisEnv()

# Define the model
model = Model(input_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Data collection (if needed)
data = []  # Example data collection

# Convert data to PyTorch tensors (Example data)
states = torch.tensor([0] * 200, dtype=torch.float32)
actions = torch.tensor([0], dtype=torch.float32)
rewards = torch.tensor([0], dtype=torch.float32)
next_states = torch.tensor([0] * 200, dtype=torch.float32)

# Define dataset and dataloader
dataset = TensorDataset(states, actions, rewards, next_states)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Model training loop
def train_model(model, criterion, optimizer, dataloader, num_epochs):
    for epoch in range(num_epochs):
        for batch in dataloader:
            state_batch, action_batch, reward_batch, next_state_batch = batch

            # Forward pass
            output = model(state_batch)

            # Compute loss
            loss = criterion(output, action_batch)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("Model training complete.")


# Call the train_model function with the required parameters
train_model(model, criterion, optimizer, dataloader, num_epochs)
