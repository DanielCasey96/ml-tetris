import gym
import numpy as np
import pygame
import random

class MyTetrisEnv(gym.Env):
    def __init__(self):
        super(MyTetrisEnv, self).__init__()
        self.game_over_flag = False
        self.cell_size = 30
        self.playable_area_width = 8
        self.playable_area_height = 20
        self.board_width = self.playable_area_width
        self.board_height = self.playable_area_height
        self.board = [[0] * self.board_width for _ in range(self.board_height)]
        self.current_piece = None
        self.current_piece_x = 3
        self.current_piece_y = 0
        self.score = 0

    def step(self, action):
        # Take action in the environment and return next state, reward, and done flag
        # Action: 0 - move left, 1 - move right, 2 - rotate, 3 - move down (soft drop)

        reward = 0
        done = False

        # Implement action handling and game logic here
        # Update the environment based on the action taken and determine the next state

        next_state = self._get_observation()  # Example: Get the next state from the observation method

        return next_state, reward, done, {}


    def reset(self):
        # Reset the environment to its initial state and return initial state
        self.game_over_flag = False
        self.board = [[0] * self.board_width for _ in range(self.board_height)]
        self.current_piece = None
        self.current_piece_x = 3
        self.current_piece_y = 0
        self.score = 0
        return self._get_observation()

    def render(self, mode='human'):
        # Visualize the current state of the environment
        if mode == 'human':
            self._render_human()
        elif mode == 'array':
            return self._render_array()

    def _get_observation(self):
        # Get the current state of the environment (observation)
        observation = {
            "board": self.board,
            "current_piece": self.current_piece,
            "current_piece_x": self.current_piece_x,
            "current_piece_y": self.current_piece_y
        }
        return observation

    def _render_human(self):
        # Render the environment using Pygame
        screen_width = self.board_width * self.cell_size
        screen_height = self.board_height * self.cell_size
        screen = pygame.Surface((screen_width, screen_height))

        # Render board
        for y in range(self.board_height):
            for x in range(self.board_width):
                if self.board[y][x] == 0:
                    pygame.draw.rect(screen, (128, 128, 128),
                                     (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
                else:
                    pygame.draw.rect(screen, (255, 255, 255),
                                     (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

        # Render current piece
        if self.current_piece:
            for y, row in enumerate(self.current_piece):
                for x, cell in enumerate(row):
                    if cell:
                        pygame.draw.rect(screen, (255, 255, 255), (
                            (x + self.current_piece_x) * self.cell_size, (y + self.current_piece_y) * self.cell_size,
                            self.cell_size, self.cell_size))

        return screen

    def _render_array(self):
        # Render the environment as a numpy array
        render_array = np.zeros((self.board_height, self.board_width), dtype=int)

        # Fill the array with the board state
        for y in range(self.board_height):
            for x in range(self.board_width):
                render_array[y][x] = self.board[y][x]

        # Add the current piece to the array
        if self.current_piece:
            for y, row in enumerate(self.current_piece):
                for x, cell in enumerate(row):
                    if cell:
                        render_array[self.current_piece_y + y][self.current_piece_x + x] = 1

        return render_array

