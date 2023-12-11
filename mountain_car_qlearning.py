

import gymnasium as gym
import numpy as np
import pickle

def train(episodes):
    env = gym.make('MountainCar-v0', render_mode=None)

    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vol_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    q_table = np.zeros((len(pos_space), len(vol_space), env.action_space.n))
    learning_rate = 0.9
    discount_factor = 0.9

    epsilon = 1
    epsilon_decay_rate = 2 / episodes
    rng = np.random.default_rng()
    # Trainning
    for episode in range(episodes):
        if episode % 100 == 0:
            print('episode:', episode)
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vol_space)
        terminated = False
        rewards = 0

        while (not terminated and rewards > -1000):
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_p, state_v, :])
            # print(env.step(action))
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vol_space)

            q_table[state_p, state_v, action] = q_table[state_p, state_v, action] + learning_rate * (
                    rewards + discount_factor * np.max(q_table[new_state_p, new_state_v, :]) - q_table[
                state_p, state_v, action]
            )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)

    env.close()
    with open('mountain_car_q.pkl','wb') as f:
        pickle.dump(q_table,f)


def test():
    # Testing
    with open('mountain_car_q.pkl', 'rb') as file:
        q_table = pickle.load(file)
    env = gym.make('MountainCar-v0', render_mode='human')
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vol_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)


    while True:
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vol_space)
        terminated = False
        rewards = 0

        while (not terminated and rewards > -1000):
            action = np.argmax(q_table[state_p, state_v, :])
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vol_space)

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            rewards += reward



def run():
    # train(5000)
    test()



if __name__ == '__main__':
    run()