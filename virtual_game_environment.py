import time
import math
import numpy as np
import gym
from gym import spaces
from utils import format_state, ACTION_NAMES



class GameEnvironment(gym.Env):
    def __init__(self, index):
        self.index = index
        self.observation_space = spaces.Box(low=0, high=2048, shape=(4, 4), dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=np.int32)
        self._place_random_tile()
        self._place_random_tile()
        return self.get_state()

    def set_state(self, state):
        self.board = np.array(state)

    def place_tile(self, i, j, value):
        self.board[i, j] = value

    def _place_random_tile(self):
        # Place a random tile (1 or 2) in an empty cell on the board
        empty_cells = [(i, j) for i in range(4) for j in range(4) if self.board[i, j] == 0]
        if empty_cells:
            i, j = empty_cells[np.random.randint(len(empty_cells))]
            self.place_tile(i, j, 1 if np.random.rand() < 0.9 else 2)
            return (i, j)

    def step(self, direction):
        current_state = self.get_state()

        score = self._move(direction)
        if score >= 0:
            self._place_random_tile()

        next_state = self.get_state()

        #return next_state, self._get_reward(current_state, next_state), self.is_game_over(), {"is_valid": True, 'score': score}
        
        # If the move is valid, place a random tile

        done = self.is_game_over()

        #if done:
        #    max = max_tile(next_state)
        #    if max > 128:
        #        print(f"Game over. Max tile: {max}")

        return next_state, self._get_reward(current_state, next_state), done, {"is_valid": score >= 0, 'score': score}

    def _state_score(self, state):
        non_zero_tiles = np.count_nonzero(state)
        score = 16 - non_zero_tiles if non_zero_tiles < 16 else -1000000

        sum = 0
        penalty = 0
        for i in range(16):
            sum += state[i // 4, i % 4]
            if i % 4 != 3:
                penalty += abs(state[i // 4, i % 4] - state[i // 4, i % 4 + 1])
            if i < 12:
                penalty += abs(state[i // 4, i % 4] - state[i // 4 + 1, i % 4])
        return score + (sum * 4 - penalty) * 2

    def _get_reward(self, current_state, next_state):
        return self._state_score(next_state) - self._state_score(current_state)

    def _move(self, direction):
        # Rotate the board if not moving left, rotate back after moving
        temp_board = rotate_board_for_direction(self.board.copy(), direction)

        # Move tiles to the left
        
        score = sum(merge_left(temp_board, row) for row in range(4))
        
        # Rotate the board back to its original orientation
        temp_board = rotate_board_for_direction(temp_board, direction)
    
        if np.array_equal(self.board, temp_board):
            # Invalid move, revert to previous state
            return -1

        # Update the game board
        self.board = temp_board

        return score

    def get_state(self):
        return self.board.copy()

    def is_game_over(self):
        # Check if the board is full (no empty cells)
        if 0 in self.board:
            # If there are empty cells, the game is not over
            return False

        # Check for adjacent tiles with the same value horizontally and vertically
        for row in range(4):
            for col in range(3):
                if self.board[row, col] == self.board[row, col + 1]:
                    return False  # Horizontal adjacent tiles with the same value

        for col in range(4):
            for row in range(3):
                if self.board[row, col] == self.board[row + 1, col]:
                    return False  # Vertical adjacent tiles with the same value

        # If no adjacent tiles with the same value are found, the game is over
        return True

    def close(self):
        pass

def adjac_score(state):
    return adjac_score_horizon(state) + adjac_score_horizon(state.T)

def adjac_score_horizon(state):
    s = 0
    state = np.log2(state+1)
    for i in range(4):
        for j in range(3):
            if state[i][j] > 0 and state[i][j+1] > 0:
                bigger = max(state[i][j], state[i][j+1])
                smaller = min(state[i][j], state[i][j+1])
                s += 10 * (smaller / bigger)
    return s

def merge_left(board, row):
    score = 0
    # find non-zero tiles in specified row
    tiles = board[row, :]
    tiles = tiles[tiles != 0]
    # merge adjacent tiles of equal value
    for i in range(len(tiles) - 1):
        if tiles[i] == tiles[i + 1]:
            tiles[i] *= 2
            tiles[i + 1] = 0
            score += tiles[i]
    tiles = tiles[tiles != 0]
    # update board with tiles
    board[row, :len(tiles)] = tiles
    board[row, len(tiles):] = 0
    return score

def rotate_board_for_direction(board, direction):
    # Rotate the board if not moving left, rotate back after moving

    if direction == 0:  # Up
        return board.T
    elif direction == 1:  # Down
        return np.flip(board.T)
    elif direction == 2:  # Left
        return board.copy()
    elif direction == 3:  # Right
        return np.flip(board, axis=1)
    
    raise ValueError(f"Invalid direction: {direction}")


def max_tile(state):
    return max(tile for row in state for tile in row)


if __name__ == "__main__":
    env = GameEnvironment()

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

            print(f"Action: {ACTION_NAMES[action]}")
            print(format_state(obs, multiLine=True))

        obs = env.reset()
        print("Reset")
        print(format_state(obs, multiLine=True))

    env.close()