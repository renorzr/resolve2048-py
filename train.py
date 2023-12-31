import os
import sys
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from virtual_game_environment import GameEnvironment

GAME_URL = 'file:///Users/reno/Downloads/1024game.html'
NUM_ENV = 24
LOG_DIR = 'logs'
STEPS = 1_000_000
os.makedirs(LOG_DIR, exist_ok=True)

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(index):
    def _init():
        return GameEnvironment(index)
    return _init

def main():
    # Set up the environment and model
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENV)])

    # Set linear schedule for learning rate
    # Start
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)

    # fine-tune
    # lr_schedule = linear_schedule(5.0e-5, 2.5e-6)

    # Set linear scheduler for clip range
    # Start
    clip_range_schedule = linear_schedule(0.15, 0.025)

    # fine-tune
    # clip_range_schedule = linear_schedule(0.075, 0.025)

    model = PPO(
        "MlpPolicy", 
        env,
        device="cuda", 
        verbose=1,
        n_steps=512,
        batch_size=512,
        n_epochs=4,
        gamma=0.94,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log="logs"
    )

    # Set the save directory
    save_dir = "trained_models_test"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_interval = 3000 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_2048")

    start_time = time.time()
    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file
    
        model.learn(
            total_timesteps=STEPS,
            callback=[checkpoint_callback]
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    print(f"Done {STEPS} steps in {time.time() - start_time} seconds")

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_2048_1M.zip"))

if __name__ == "__main__":
    main()
