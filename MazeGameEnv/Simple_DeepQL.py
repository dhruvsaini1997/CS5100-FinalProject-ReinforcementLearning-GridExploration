import numpy as np
import torch
from MazeGame import MazeGame
import random
from TestModel import TestModel
from IPython.display import clear_output
from matplotlib import pylab as plt

# Define neural network architecture
model = torch.nn.Sequential(
    torch.nn.Linear(64, 150),
    torch.nn.ReLU(),
    torch.nn.Linear(150, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 4)
)

# Define loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Define parameters for Q-learning
gamma = 0.9
epsilon = 1.0
action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

# Training loop
epochs = 1000
losses = []  # List to store losses
for epoch in range(epochs):
    # Initialize game environment
    game = MazeGame(size=4, mode='static')
    state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0).float()
    status = 1
    
    # Episode loop
    while status == 1:
        # Epsilon-greedy policy
        q_values = model(state)
        if random.random() < epsilon:
            action = np.random.randint(0, 4)
        else:
            action = torch.argmax(q_values).item()
        
        # Take action in the game
        game.make_move(action_set[action])
        next_state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0).float()
        reward = game.reward()
        
        # Update Q-values
        with torch.no_grad():
            next_q_values = model(next_state)
            max_q_value = torch.max(next_q_values)
            target_q_value = reward + gamma * max_q_value
            target_q_value = torch.Tensor([target_q_value]).detach()
        
        current_q_value = q_values.squeeze()[action]
        loss = loss_fn(current_q_value, target_q_value)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update state
        state = next_state
        
        # Check if game is over
        if reward != -1:
            status = 0
    
    # Decrease epsilon
    if epsilon > 0.1:
        epsilon -= 1 / epochs
    
    # Print and store loss
    print(f'Epoch: {epoch}, Loss: {loss.item()}')
    clear_output(wait=True)
    losses.append(loss.item())

# Plot loss
plt.figure(figsize=(10, 7))
plt.plot(losses)
plt.xlabel('Epochs', fontsize=22)
plt.ylabel('Loss', fontsize=22)
plt.savefig('MazeGameEnv/Results/Vanilla_Model_Episode_VS_Loss_plot.png', bbox_inches='tight', pad_inches=0)
plt.close()

# Test the model
max_games = 1000
wins = 0
tester = TestModel()
for _ in range(max_games):
    if tester.test_model(model, mode='random', display=False):
        wins += 1

win_percentage = wins / max_games * 100
print(f'Games played: {max_games}, # of wins: {wins}')
print(f'Win percentage: {win_percentage:.2f}%')

# Calculate accuracy
accuracy = wins / max_games * 100

# Calculate incorrect predictions percentage
incorrect_percentage = 100 - accuracy

# Create labels and sizes for the pie chart
labels = ['Correct', 'Incorrect']
sizes = [accuracy, incorrect_percentage]

# Define colors for the pie chart
colors = ['lightcoral', 'lightskyblue']

# Create the pie chart
plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Vanilla Deep QLearning Model Accuracy')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
