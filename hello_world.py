import gym

env = gym.make('CartPole-v1') #,render_mode="human")
observation = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # Random action for illustration purposes
    observation, reward, done, _ = env.step(action)
    # print(a)

    if done:
        observation = env.reset()

env.close()
