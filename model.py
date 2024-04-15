import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tetris import TetrisGame


# Define the neural network model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return x


# Define hyperparameters
input_size = 200
output_size = 1
learning_rate = 0.001
data_length = 100
batch_size = 32

# Convert data to PyTorch tensors (Example data)
states = torch.zeros((data_length, 200), dtype=torch.float32)
actions = torch.zeros((data_length, 1), dtype=torch.float32)
rewards = torch.zeros((data_length, 1), dtype=torch.float32)
next_states = torch.zeros((data_length, 200), dtype=torch.float32)

# Define dataset and dataloader
dataset = TensorDataset(states, actions, rewards, next_states)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = Model()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Model training loop
# Model training loop
def train_model(model, criterion, optimizer, dataloader, num_epochs):
    prev_loss = float('inf')  # Initialize previous loss with infinity

    for epoch in range(num_epochs):
        epoch_loss = 0.0
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

            epoch_loss += loss.item()

        # Print epoch loss
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")

        # Calculate and print improvement
        improvement = prev_loss - (epoch_loss / len(dataloader))
        print(f"Improvement: {improvement}")

        # Update previous loss for the next epoch
        prev_loss = epoch_loss / len(dataloader)

        # Initialize Tetris game environment
        game = TetrisGame(model)

        # Run Tetris game using the trained model
        game.run()

    print("Model training complete.")



