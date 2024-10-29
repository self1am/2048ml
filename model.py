import numpy as np
import tensorflow as tf
from collections import deque
import random
import copy

class Game2048Env:
    def __init__(self):
        self.board_size = 4
        self.reset()
        
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.score = 0
        self._add_tile()
        self._add_tile()
        return self._get_state()
    
    def _add_tile(self):
        empty_cells = [(i, j) for i in range(self.board_size) 
                      for j in range(self.board_size) if self.board[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4
    
    def _get_state(self):
        # Convert board to feature representation
        state = np.log2(np.where(self.board > 0, self.board, 1)).reshape(-1)
        return state
    
    def step(self, action):
        old_board = self.board.copy()
        old_score = self.score
        
        moved = self._move(action)
        if moved:
            self._add_tile()
        
        done = self._is_game_over()
        reward = self._calculate_reward(old_board, old_score)
        
        return self._get_state(), reward, done
    
    def _move(self, action):
        # 0: up, 1: right, 2: down, 3: left
        moved = False
        board = self.board
        
        if action in [0, 2]:  # up or down
            for j in range(self.board_size):
                column = board[:, j]
                if action == 2:  # down
                    column = column[::-1]
                
                # Remove zeros and merge
                column = column[column != 0]
                for i in range(len(column) - 1):
                    if column[i] == column[i + 1]:
                        column[i] *= 2
                        self.score += column[i]
                        column = np.delete(column, i + 1)
                        moved = True
                
                # Pad with zeros
                column = np.pad(column, (0, self.board_size - len(column)), 
                              'constant')
                if action == 2:  # down
                    column = column[::-1]
                
                if not np.array_equal(board[:, j], column):
                    moved = True
                board[:, j] = column
                
        else:  # left or right
            for i in range(self.board_size):
                row = board[i, :]
                if action == 1:  # right
                    row = row[::-1]
                
                # Remove zeros and merge
                row = row[row != 0]
                for j in range(len(row) - 1):
                    if row[j] == row[j + 1]:
                        row[j] *= 2
                        self.score += row[j]
                        row = np.delete(row, j + 1)
                        moved = True
                
                # Pad with zeros
                row = np.pad(row, (0, self.board_size - len(row)), 
                           'constant')
                if action == 1:  # right
                    row = row[::-1]
                
                if not np.array_equal(board[i, :], row):
                    moved = True
                board[i, :] = row
        
        return moved
    
    def _is_game_over(self):
        if 0 in self.board:
            return False
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i < self.board_size - 1 and 
                    self.board[i][j] == self.board[i + 1][j]) or \
                   (j < self.board_size - 1 and 
                    self.board[i][j] == self.board[i][j + 1]):
                    return False
        return True
    
    def _calculate_reward(self, old_board, old_score):
        # Reward for score increase
        score_reward = (self.score - old_score) / 100.0
        
        # Penalty for game over
        if self._is_game_over():
            return -1.0
        
        # Reward for keeping high values in corners
        corner_reward = 0.0
        corners = [(0,0), (0,3), (3,0), (3,3)]
        for i, j in corners:
            if self.board[i][j] >= 8:
                corner_reward += np.log2(self.board[i][j]) * 0.1
        
        return score_reward + corner_reward

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=self.state_size, 
                                activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse',
                     optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        targets = rewards + self.gamma * \
                 np.amax(self.model.predict(next_states), axis=1) * (1 - dones)
        target_f = self.model.predict(states)
        target_f[np.arange(batch_size), actions] = targets
        
        self.model.fit(states, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent():
    env = Game2048Env()
    state_size = 16  # 4x4 board flattened
    action_size = 4  # up, right, down, left
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 1000
    
    best_score = 0
    scores = []
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            if done:
                scores.append(env.score)
                if env.score > best_score:
                    best_score = env.score
                    # Save the model if it achieved a new high score
                    agent.model.save('2048_best_model.h5')
                
                print(f"Episode: {e+1}/{episodes}, Score: {env.score}, " \
                      f"Best: {best_score}, Epsilon: {agent.epsilon:.2f}")
                break
    
    return agent, scores

# Function to use the trained model
def play_game(model_path='2048_best_model.h5'):
    env = Game2048Env()
    model = tf.keras.models.load_model(model_path)
    
    state = env.reset()
    done = False
    total_score = 0
    
    while not done:
        act_values = model.predict(state.reshape(1, -1))
        action = np.argmax(act_values[0])
        state, reward, done = env.step(action)
        total_score += reward
        
        # Print the board for visualization
        print(env.board)
        print(f"Score: {env.score}")
        
    return total_score

if __name__ == "__main__":
    # Train the agent
    agent, training_scores = train_agent()
    
    # Plot training progress
    import matplotlib.pyplot as plt
    plt.plot(training_scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()
    
    # Play a game with the trained model
    final_score = play_game()
    print(f"Final Score: {final_score}")