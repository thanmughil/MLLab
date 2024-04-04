import numpy as np

env = np.array([[-1, -1, 0],
                [-1, -1, 0],
                [0, 0, 10]])

Q = np.zeros((3, 3))
learning_rate = 0.8
discount_factor = 0.9
epochs = 1000
epsilon = 0.1

for _ in range(epochs):
    current_state = np.random.randint(0, 3)
    
    while current_state != 2:
        if np.random.rand() < epsilon:
            action = np.random.choice(np.where(env[current_state] >= 0)[0])
        else:
            action = np.argmax(Q[current_state])
        
        next_state = action
        reward = env[current_state, action]
        
        Q[current_state, action] = (1 - learning_rate) * Q[current_state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state]))
        
        current_state = next_state

print("Learned Q-values:")
print(Q)

current_state = 0  
path = [current_state]

while current_state != 2:
    action = np.argmax(Q[current_state])
    current_state = action
    path.append(current_state)

print("Optimal path:", path)
