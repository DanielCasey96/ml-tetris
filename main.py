import pygame
from model import train_model, dataloader, optimizer, criterion, model
from tetris import TetrisGame

pygame.font.init()


def main():
    num_epochs = 10000

    # Train the model
    train_model(model, criterion, optimizer, dataloader, num_epochs)

    # Play Tetris using the trained model
    game = TetrisGame(model)
    game.run()


if __name__ == "__main__":
    main()
