import gymnasium as gym
import numpy as np

def discretize_state(state, bins):
    """Discretize a continuous state into bins."""
    state_min, state_max = env.observation_space.low, env.observation_space.high
    discretized_state = [np.digitize(s, np.linspace(state_min[i], state_max[i], bins + 1)[1:-1]) for i, s in enumerate(state)]
    return tuple(discretized_state)

# Create CartPole environment
env = gym.make('CartPole-v1')

# Discretize the state space
num_bins = 200  # You can adjust this parameter based on the granularity needed
num_states = tuple([num_bins] * env.observation_space.shape[0])
num_actions = env.action_space.n
q_table = np.zeros(num_states + (num_actions,))  # Adjust the shape of the Q-table

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
num_episodes = 10000

# Q-learning algorithm
for episode in range(num_episodes):
    if episode%100 == 1:
        print('Episode:',episode)
    state = env.reset()
    state_discrete = discretize_state(state, num_bins)

    for t in range(10000):  # Limiting the number of time steps for safety
        # Epsilon-greedy exploration strategy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state_discrete])  # Exploit

        # Take the chosen action
        next_state, reward, done, _ = env.step(action)
        next_state_discrete = discretize_state(next_state, num_bins)

        # Q-value update using the Bellman equation
        q_table[state_discrete + (action,)] += learning_rate * (
            reward + discount_factor * np.max(q_table[next_state_discrete]) - q_table[state_discrete + (action,)]
        )

        state_discrete = next_state_discrete

        if done:
            break

# Use the learned policy to play CartPole
state = env.reset()
state_discrete = discretize_state(state, num_bins)

while True:
    action = np.argmax(q_table[state_discrete])  # Choose the best action
    next_state, _, done, _ = env.step(action)
    env.render()

    next_state_discrete = discretize_state(next_state, num_bins)
    state_discrete = next_state_discrete

    if done:
        env.reset()

env.close()
