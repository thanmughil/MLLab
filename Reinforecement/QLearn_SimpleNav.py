import numpy as np

env = np.array([[-1, -2, -2],
                [-0.75, -0.2, -0.5],
                [-0.5, -0.2,  1]])

Q = np.zeros((9, 4))

alpha = 0.1
gamma = 0.9
epsilon = 1.0

def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice([0, 1, 2, 3])
    else:
        return np.argmax(Q[state, :])

def update_Q(state, action, reward, next_state):
    Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

def q_learning():
    global epsilon
    for episode in range(1000):
        state = 0
        done = False
        while not done:
            action = choose_action(state)
            next_state = state
            
            if action == 0:
                next_state = max(0, state - 3)
            elif action == 1:
                next_state = min(8, state + 1)
            elif action == 2:
                next_state = min(8, state + 3)
            elif action == 3:
                next_state = max(0, state - 1)
                
            reward = env[int(next_state / 3), next_state % 3]
            update_Q(state, action, reward, next_state)
            state = next_state
            
            if reward == 1:
                done = True
                break
    
    return state

def get_optimal_path(start_state):
    state = start_state
    path = [state]
    while True:
        action = np.argmax(Q[state, :])
        next_state = state
        if action == 0:
            next_state = max(0, state - 3)
        elif action == 1:
            next_state = min(8, state + 1)
        elif action == 2:
            next_state = min(8, state + 3)
        elif action == 3:
            next_state = max(0, state - 1)
        path.append(next_state)
        state = next_state
        if env[int(state / 3), state % 3] == 1:
            break
    return path

goal_state = q_learning()
optimal_path = get_optimal_path(0)

print("Optimal path taken by the agent:")
for state in optimal_path:
    print(f"State: {state}, Coordinate: ({state // 3}, {state % 3})")