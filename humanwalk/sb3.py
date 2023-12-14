
import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C
import os
import argparse

model_dir = 'models'
log_dir = 'logs'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def train(env, sb3_algo):
    match sb3_algo:
        case 'SAC':
            model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'TD3':
            model = TD3('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'A2C':
            model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case _:
            print('Algorithm not found')
            return
    TIMESTPES = 25000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTPES,reset_num_timesteps=False)
        model.save(f'{model_dir}/{sb3_algo}_{TIMESTPES*iters}')

def test(env, sb3_algo, path_to_model):
    match sb3_algo:
        case 'SAC':
            model = SAC.load(path_to_model, env)
        case 'TD3':
            model = TD3.load(path_to_model, env)
        case 'A2C':
            model = A2C.load(path_to_model, env)
        case _:
            print('Algorithm not found')
            return

    while(True):
        obs = env.reset()[0]
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _, _ = env.step(action)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium enviroment i.e. Humanoid-v4')
    parser.add_argument('sb3_algo')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', metavar='path_to_model')
    args = parser.parse_args()

    if args.train:
        env = gym.make(args.gymenv, render_mode=None)
        train(env, args.sb3_algo)
    
    #return

    if args.test:
        if os.path.isfile(args.test):
            env = gym.make(args.gymenv, render_mode='human')
            test(env, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found')

