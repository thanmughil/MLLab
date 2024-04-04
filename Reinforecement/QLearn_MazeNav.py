import numpy as np

class GridEnvironment:
    def __init__(self, grid_size, obstacles):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size, dtype=int) 
        self.start_state = (0, 0)
        self.goal_state = (0, grid_size[1] - 1)
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  
        for obstacle in obstacles:
            self.grid[obstacle] = 1 
    def step(self, state, action):
        new_state = (state[0] + action[0], state[1] + action[1])
        if self.is_valid_state(new_state):
            return new_state
        else:
            return state
    
    def is_valid_state(self, state):
        x, y = state
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and self.grid[state] != 1
    
    def is_goal_state(self, state):
        return state == self.goal_state
    
    def visualize_path(self, path):
        grid_copy = self.grid.tolist()
        for state in path:
            x, y = state
            grid_copy[x][y] = 'X'  
        print("Grid with path:")
        for row in grid_copy:
            print(' '.join(map(str, row)))


def train_q_learning(grid_size, obstacles, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    env = GridEnvironment(grid_size, obstacles)
    q_table = np.zeros(env.grid_size + (len(env.actions),))
    for _ in range(num_episodes):
        state = env.start_state
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(len(env.actions))
            else:
                action = np.argmax(q_table[state])
            next_state = env.step(state, env.actions[action])
            if env.is_goal_state(next_state):
                reward = 1
                done = True
            else:
                reward = 0
            q_table[state + (action,)] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state + (action,)])
            state = next_state
    return q_table

def test_q_learning(grid_size, obstacles, q_table):
    env = GridEnvironment(grid_size, obstacles)
    state = env.start_state
    path = [state]
    done = False
    while not done:
        action = np.argmax(q_table[state])
        next_state = env.step(state, env.actions[action])
        path.append(next_state)
        state = next_state
        if env.is_goal_state(next_state):
            done = True
    return path

rows = int(input("Enter the number of rows: "))
cols = int(input("Enter the number of columns: "))
grid_size = (rows, cols)


grid = np.arange(1, rows * cols + 1).reshape(grid_size)

print("Grid with numbers:")
for row in grid:
    print(' '.join(map(str, row)))

obstacles = []
num_obstacles = int(input("Enter the number of obstacles: "))
for i in range(num_obstacles):
    obstacle_spot = int(input(f"Choose a spot for obstacle {i + 1} (enter the number): "))
    obstacle_row = (obstacle_spot - 1) // cols
    obstacle_col = (obstacle_spot - 1) % cols
    obstacles.append((obstacle_row, obstacle_col))

q_table = train_q_learning(grid_size, obstacles)

path = test_q_learning(grid_size, obstacles, q_table)

env = GridEnvironment(grid_size, obstacles)
env.visualize_path(path)