import gymnasium as gym
import numpy as np
import pickle
from gym.envs.toy_text.frozen_lake import generate_random_map


def run(episodes):
    #random_map = generate_random_map(size=8)
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False) #,render_mode='human') map_name='16x16'  desc = random_map
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    learning_rate = 0.1
    discounted_factor = 0.99

    epsilon = 1
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()


    for i in range(episodes):

        if i%100 == 0:
            print('Episode:',i)

        state = env.reset()[0]
        terminated = False
        truncated = False

        while(not terminated and not truncated):
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state,:])
            new_state, reward, terminated, truncated,_ = env.step(action)
            # env.render()
            q_table[state,action] = q_table[state,action] + learning_rate * (reward + discounted_factor*np.max(q_table[new_state,:]) - q_table[state,action])

            state = new_state

        epsilon  = max(epsilon-epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate = 0.0001

    env.close()


    '''f = open('frozen_lake8x8.pkl','wb')
    pickle.dump(q_table, f)
    f.close()'''

    ## Disp final path
    env = gym.make('FrozenLake-v1',map_name='8x8',is_slippery=False,render_mode='human')
    state = env.reset()[0]
    print(q_table)
    while True:
        action = np.argmax(q_table[state, :])
        # print(action)
        new_state, reward, terminated, truncated, _ = env.step(action)
        #env.render()
        state = new_state

        if terminated:
            state = env.reset()[0]

    env.close()


if __name__ == '__main__':
    run(15000)