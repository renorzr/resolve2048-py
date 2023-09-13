import time
import numpy as np
import gym
from gym import spaces
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


ARROW_KEYS = (Keys.ARROW_UP, Keys.ARROW_DOWN, Keys.ARROW_LEFT, Keys.ARROW_RIGHT)

class GameEnvironment(gym.Env):
    def __init__(self, index, game_url, headless=False):
        self.index = index
        # Define the action space (e.g., up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Define the observation space (e.g., game board state)
        self.observation_space = spaces.Box(low=0, high=2048, shape=(4, 4), dtype=np.int32)

        # Initialize the Selenium web driver and open the game URL
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--start-maximized")

        self.driver = webdriver.Chrome(chrome_options)
        self.driver.get(game_url)

    def reset(self):
        # Reset the game by refreshing the page
        self.driver.refresh()
        time.sleep(1)  # Allow time for the game to load
        return self.get_state()

    def step(self, action):
        current_state = self.get_state()

        self.driver.find_element(By.TAG_NAME, 'body').send_keys(ARROW_KEYS[action])

        # Add a small delay to allow the game to update
        time.sleep(0.5)  # You may need to adjust the delay based on your game's responsiveness

        next_state = self.get_state()
        reward = self.get_reward(current_state, next_state)
        done = self.is_game_over()

        
        return next_state, reward, done, {}

    def get_state(self):
        while True:
            try:
                # Locate the tile container element
                tile_container = self.driver.find_element(By.CLASS_NAME, "tile-container")
                
                # Get all tile elements within the container
                tile_elements = tile_container.find_elements(By.CLASS_NAME, "tile")
                
                # Initialize an empty game state matrix (4x4 grid)
                game_state = [[0] * 4 for _ in range(4)]
                
                for tile_element in tile_elements:
                    # Parse the tile's position and value
                    classes = tile_element.get_attribute("class").split()
                    position_class = [cls for cls in classes if cls.startswith("tile-position")]
                    value = int(tile_element.find_element(By.CLASS_NAME, "tile-inner").text)
                    
                    if position_class:
                        # Extract row and column information from the class
                        position_class = position_class[0]
                        _, _, col, row = position_class.split("-")
                        row = int(row) - 1  # Adjust for 0-based indexing
                        col = int(col) - 1  # Adjust for 0-based indexing
                        
                        # Set the tile's value in the game state matrix
                        game_state[row][col] = value
                
                return game_state
            except:
                print("Error getting game state, retrying...")
                time.sleep(0.1)

    def get_reward(self, current_state, next_state):
        # Negative reward for game over
        if self.is_game_over():
            return 1024 - max_tile(next_state)

        # Penalties for invalid moves (negative reward)
        if current_state == next_state:
            print("Invalid move")
            return -2
        
        return 1

    def is_game_over(self):
        # Detect if the game is over (e.g., check for a game over screen)
        try:
            retry_button = self.driver.find_element(By.CLASS_NAME, "retry-button")
            return retry_button.is_displayed()
        except NoSuchElementException:
            return False  # Game is not over if the retry button is not found

def max_tile(state):
    return max(tile for row in state for tile in row)


if __name__ == "__main__":
    from utils import format_state
    env = GameEnvironment(1, 'file:///Users/reno/Downloads/1024game.html', False)

    num_episodes = 1
    episode_reward_sum = 0
    num_victory = 0
    obs = env.get_state()
    print("Start")
    print(format_state(obs, multiLine=True))

    for _ in range(num_episodes):
        done = False
        
        total_reward = 0

        while not done:
            timestamp = time.time()

            action = np.random.randint(4)
            obs, reward, done, info = env.step(action)

            print(f"Action: {action}")
            print(format_state(obs, multiLine=True))

        obs = env.reset()
        print("Reset")
        print(format_state(obs, multiLine=True))

    env.close()