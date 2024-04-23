import numpy as np

class Piece:
    def __init__(self, name, code, position=(0, 0)):
        """
        Represents a piece on the board.
        
        Parameters:
            name (str): Name of the piece.
            code (str): Code representing the piece on the board (usually an ASCII character).
            position (tuple): Position of the piece on the board, default is (0, 0).
        """
        self.name = name
        self.code = code
        self.position = position

class MazeBoard:
    EMPTY_SPACE = ' '

    def __init__(self, size=4):
        """
        Represents the game board.
        
        Parameters:
            size (int): Size of the square board, default is 4.
        """
        self.size = size
        self.components = {}  # Dictionary to store board pieces
        self.obstacles = {}   # Dictionary to store obstacles

    def add_piece(self, name, code, position=(0, 0)):
        """
        Add a piece to the board.
        
        Parameters:
            name (str): Name of the piece.
            code (str): Code representing the piece on the board.
            position (tuple): Position of the piece on the board, default is (0, 0).
        """
        self.components[name] = Piece(name, code, position)


    def move_piece(self, name, position):
        """
        Move a piece to a new position on the board.
        
        Parameters:
            name (str): Name of the piece to be moved.
            position (tuple): New position of the piece on the board.
        """
        # Check if the new position is not blocked by any obstacle
        if all(position not in obstacle for obstacle, _ in self.obstacles.values()):
            self.components[name].position = position

    def render(self):
        """
        Render the current state of the board as a 2D array of characters.
        """
        board = np.full((self.size, self.size), self.EMPTY_SPACE, dtype='<U2')
        # Render components (pieces)
        for piece in self.components.values():
            board[piece.position] = piece.code
        # Render obstacles (obstacles)
        for obstacle, code in self.obstacles.values():
            board[obstacle] = code
        return board

    def render_np(self):
        """
        Render the current state of the board as a numpy array with components represented by 1s and empty spaces by 0s.
        """
        board = np.zeros((len(self.components) + len(self.obstacles), self.size, self.size), dtype=np.uint8)
        # Render components as 1s in the numpy array
        for i, piece in enumerate(self.components.values()):
            board[i, piece.position[0], piece.position[1]] = 1
        # Render obstacles as 1s in the numpy array
        for i, (obstacle, _) in enumerate(self.obstacles.values(), start=len(self.components)):
            board[i, obstacle[0], obstacle[1]] = 1
        return board
