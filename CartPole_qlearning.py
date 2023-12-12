

import gymnasium as gym
import numpy as np
import pickle

def train():
    env = gym.make('CartPole-v1', render_mode=None)

    pos_space = np.linspace(-2.4, 2.4, 10)
    vol_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-0.2095, 0.2095, 10)
    ang_vol_space = np.linspace(-4, 4, 10)

    q_table = np.zeros((len(pos_space)+1, len(vol_space)+1, len(ang_space)+1, len(ang_vol_space)+1, env.action_space.n))
    learning_rate = 0.1
    discount_factor = 0.99

    epsilon = 1
    epsilon_decay_rate = 0.00001
    rng = np.random.default_rng()

    rewards_per_episode = []
    i = 0

    # Trainning
    #for episode in range(episodes):
    while True:
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vol_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vol_space)
        terminated = False
        rewards = 0

        while (not terminated and rewards < 10000):
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_p, state_v, state_a, state_av, :])
            # print(env.step(action))
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vol_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vol_space)

            q_table[state_p, state_v, state_a, state_av, action] = q_table[state_p, state_v, state_a, state_av, action] + learning_rate * (
                    rewards + discount_factor * np.max(q_table[new_state_p, new_state_v, new_state_a, new_state_av, :]) - q_table[
                state_p, state_v, state_a, state_av, action]
            )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av
            rewards += reward

        rewards_per_episode.append(rewards)

        mean_rewards = np.mean(rewards_per_episode[-100:])
        if i%100 == 0:
            print(f'Episode: {i} {rewards} Epsilon: {epsilon:0.2f} Mean Rewards {mean_rewards:0.1f}')
        if mean_rewards > 1000:
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        i += 1

    env.close()
    with open('CartPole_q.pkl','wb') as f:
        pickle.dump(q_table,f)


def test():
    # Testing
    with open('CartPole_q.pkl', 'rb') as file:
        q_table = pickle.load(file)
    env = gym.make('CartPole-v1', render_mode='human')
    pos_space = np.linspace(-2.4, 2.4, 10)
    vol_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-0.2095, 0.2095, 10)
    ang_vol_space = np.linspace(-4, 4, 10)

    while True:
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vol_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vol_space)
        terminated = False
        rewards = 0

        while (not terminated):
            action = np.argmax(q_table[state_p, state_v, state_a, state_av, :])
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vol_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vol_space)

            #state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av
            rewards += reward
        print('Rewards:', rewards)



def run():
    train()
    test()



if __name__ == '__main__':
    run()