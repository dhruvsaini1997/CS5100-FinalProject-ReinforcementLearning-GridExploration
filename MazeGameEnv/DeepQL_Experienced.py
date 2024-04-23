import numpy as np
import torch
from MazeGame import MazeGame
from IPython.display import clear_output
import random
from collections import deque
from matplotlib import pylab as plt
from TestModel import TestModel

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
epsilon = 0.3
action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

# Training parameters
epochs = 5000
mem_size = 1000
batch_size = 200
max_moves = 50
losses = []

# Initialize experience replay memory
replay = deque(maxlen=mem_size)

# Main training loop
for epoch in range(epochs):
    # Initialize environment
    game = MazeGame(size=4, mode='random')
    state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0).float()
    mov = 0
    
    # Episode loop
    while True:
        mov += 1
        
        # Epsilon-greedy policy
        q_values = model(state)
        if random.random() < epsilon:
            action = np.random.randint(0, 4)
        else:
            action = torch.argmax(q_values).item()
        
        # Take action in the environment
        game.make_move(action_set[action])
        next_state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0).float()
        reward = game.reward()
        done = True if reward > 0 else False
        
        # Store experience in replay memory
        exp = (state, action, reward, next_state, done)
        replay.append(exp)
        state = next_state
        
        # Sample minibatch and perform Q-learning update
        if len(replay) > batch_size:
            minibatch = random.sample(replay, batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)
            
            states_batch = torch.cat(states)
            actions_batch = torch.Tensor(actions)
            rewards_batch = torch.Tensor(rewards)
            next_states_batch = torch.cat(next_states)
            dones_batch = torch.Tensor(dones)
            
            Q1 = model(states_batch)
            with torch.no_grad():
                Q2 = model(next_states_batch)
            
            Y = rewards_batch + gamma * ((1 - dones_batch) * torch.max(Q2, dim=1)[0])
            X = Q1.gather(dim=1, index=actions_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X, Y.detach())
            
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        
        # Check if episode is finished
        if reward != -1 or mov > max_moves:
            break

# Plot loss
plt.figure(figsize=(10, 7))
plt.plot(losses)
plt.xlabel('Epochs', fontsize=22)
plt.ylabel('Loss', fontsize=22)
plt.savefig('MazeGameEnv/Results/Experienced_Model_Episode_VS_Loss_plot.png', bbox_inches='tight', pad_inches=0)
plt.close()

# Test the model
tester = TestModel()
max_games = 1000
wins = sum(tester.test_model(model, mode='random', display=False) for _ in range(max_games))
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
plt.title('Deep QLearning with Experienced Replay Buffer Model Accuracy')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
