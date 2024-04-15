from model import Model, train_model
from tetris import TetrisGame
import torch


def main():
    # Define hyperparameters
    input_size = 200
    output_size = 1
    learning_rate = 0.001
    num_epochs = 1000
    batch_size = 32

    # Initialize the model
    model = Model(input_size, output_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, criterion, optimizer, num_epochs, batch_size)

    # Play Tetris using the trained model
    game = TetrisGame(model)
    game.run()


if __name__ == "__main__":
    main()
