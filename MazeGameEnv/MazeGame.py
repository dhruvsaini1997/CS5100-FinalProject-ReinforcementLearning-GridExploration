import numpy as np
from MazeBoard import MazeBoard

class MazeGame:
    def __init__(self, size=4, mode='static'):
        # Initialize MazeBoard with specified size
        self.board = MazeBoard(max(size, 4))
        # Initialize the board based on the chosen mode
        self.init_board(mode)

    def init_board(self, mode):
        # Add pieces to the board with initial positions
        self.board.add_piece('Player', 'P')
        self.board.add_piece('Goal', '+')
        self.board.add_piece('Pit', '-')
        self.board.add_piece('Wall', 'W')

        # Initialize the grid based on the chosen mode
        if mode == 'static':
            self.init_grid_static()
        elif mode == 'player':
            self.init_grid_player()
        else:
            self.init_grid_rand()

    def init_grid_static(self):
        # Set static positions for pieces
        self.board.components['Player'].position = (0, 3)
        self.board.components['Goal'].position = (0, 0)
        self.board.components['Pit'].position = (0, 1)
        self.board.components['Wall'].position = (1, 1)

    def validate_board(self):
        # Check if the board is initialized appropriately
        positions = [piece.position for piece in self.board.components.values()]
        return len(positions) == len(set(positions))

    def init_grid_player(self):
        # Initialize the grid with the player's position chosen randomly
        self.init_grid_static()
        self.board.components['Player'].position = rand_pair(0, self.board.size)
        # If the board is not valid, reinitialize
        if not self.validate_board():
            self.init_grid_player()

    def init_grid_rand(self):
        # Initialize the grid with all pieces placed randomly
        for name in ['Player', 'Goal', 'Pit', 'Wall']:
            self.board.components[name].position = rand_pair(0, self.board.size)
        # If the board is not valid, reinitialize
        if not self.validate_board():
            self.init_grid_rand()

    def validate_move(self, piece, addposition=(0, 0)):
        # Validate if a move is valid for a given piece
        new_position = add_tuple(self.board.components[piece].position, addposition)
        if new_position == self.board.components['Wall'].position:
            return 1  # Move blocked by wall
        if max(new_position) > (self.board.size - 1) or min(new_position) < 0:
            return 1  # Move outside board boundaries
        if new_position == self.board.components['Pit'].position:
            return 2  # Move leads to pit (lose game)
        return 0  # Move is valid

    def make_move(self, action):
        # Make a move based on the given action ('u', 'd', 'l', 'r')
        directions = {'u': (-1, 0), 'd': (1, 0), 'l': (0, -1), 'r': (0, 1)}
        add_position = directions.get(action)
        if add_position:
            if self.validate_move('Player', add_position) in [0, 2]:
                new_position = add_tuple(self.board.components['Player'].position, add_position)
                self.board.move_piece('Player', new_position)

    def reward(self):
        # Calculate the reward based on the player's position
        player_position = self.board.components['Player'].position
        if player_position == self.board.components['Pit'].position:
            return -10  # Player fell into pit (negative reward)
        elif player_position == self.board.components['Goal'].position:
            return 10  # Player reached the goal (positive reward)
        else:
            return -1  # No special event (small negative reward)

    def display(self):
        # Display the current state of the board
        return self.board.render()

def rand_pair(s, e):
    # Generate a random pair of integers within the given range
    return np.random.randint(s, e), np.random.randint(s, e)

def add_tuple(a, b):
    # Add corresponding elements of two tuples
    return tuple(sum(x) for x in zip(a, b))
