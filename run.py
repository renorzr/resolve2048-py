import random
import sys
import os
import time 
from utils import format_state, ACTION_NAMES

from stable_baselines3 import PPO

from virtual_game_environment import GameEnvironment, max_tile

MODEL_DIR = r"trained_models_test/"
MODEL_NAME = sys.argv[1] # e.g.: ppo_2048_10M

env = GameEnvironment(0)
model = None if MODEL_NAME == 'random' else PPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env)
done = False

num_episodes = 100
episode_reward_sum = 0
num_victory = 0
num_invalid_moves = 0
moves = 0
obs = env.get_state()

WIN_CRITERIA = 1024
sum_max_tiles = 0
sum_scores = 0

for i in range(num_episodes):
    done = False
    
    total_reward = 0

    steps = 0
    episode_score = 0
    while not done:
        timestamp = time.time()

        if MODEL_NAME == 'random':
          action = random.randint(0, 3)
        else:
          action, states = model.predict(obs)

        obs, reward, done, info = env.step(action)
        if not info['is_valid']:
            num_invalid_moves += 1

        sum_scores += info['score']
        episode_score += info['score']
        steps += 1
        moves += 1
        print(f"Epsiode {i} step {steps} score {episode_score}", end="\r")

    print(f"Episode {i} score: {episode_score} max tile: {max_tile(obs)}   ", end="\r")
    if i % 10 == 0:
        print()

    max = max_tile(obs)
    sum_max_tiles += max
    if max > WIN_CRITERIA:
        num_victory += 1

    obs = env.reset()

print(f"Average max tile: {sum_max_tiles / num_episodes}")
print(f"Average score: {sum_scores / num_episodes}")
print(f"Number of victories: {num_victory}")
print(f"Number of invalid moves: {num_invalid_moves} ({num_invalid_moves / moves * 100}%)")

env.close()