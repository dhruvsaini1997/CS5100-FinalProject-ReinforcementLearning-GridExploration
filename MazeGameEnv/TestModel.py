from MazeGame import MazeGame 
import torch
import numpy as np

class TestModel:
    # Dictionary mapping action indices to action strings
    action_set = {
        0: 'u',  # Up
        1: 'd',  # Down
        2: 'l',  # Left
        3: 'r',  # Right
    }

    def test_model(self, model, mode='static', display=True):
        # Initialize variables
        i = 0
        test_game = MazeGame(mode=mode)  # Create a MazeGame instance
        # Get initial state of the game board and add some noise for exploration
        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.from_numpy(state_).float()

        # Display initial state if required
        if display:
            print("Initial State:")
            print(test_game.display())
        
        status = 1  # Status indicating game ongoing
        while status == 1:
            # Get Q-values from the model for the current state
            qval = model(state)
            qval_ = qval.data.numpy()
            action_ = np.argmax(qval_)
            action = self.action_set[action_]  # Translate action index to action string
            
            # Display the action being taken if required
            if display:
                print('Move #: %s; Taking action: %s' % (i, action))
            
            # Make the move in the game
            test_game.make_move(action)
            
            # Get the new state of the game board and add noise
            state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            state = torch.from_numpy(state_).float()
            
            # Display the updated game board if required
            if display:
                print(test_game.display())
            
            # Check if the game has ended and update status accordingly
            reward = test_game.reward()
            if reward != -1:
                if reward > 0:
                    status = 2  # Game won
                    if display:
                        print("Game won! Reward: %s" % (reward,))
                else:
                    status = 0  # Game lost
                    if display:
                        print("Game LOST. Reward: %s" % (reward,))
            i += 1
            if i > 15:
                if display:
                    print("Game lost; too many moves.")
                break

        # Return whether the game was won or lost
        win = True if status == 2 else False
        return win
