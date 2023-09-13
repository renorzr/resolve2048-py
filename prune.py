import numpy as np
from virtual_game_environment import GameEnvironment, max_tile
from game_environment import GameEnvironment as WebDriverGameEnvironment
from utils import format_state, ACTION_NAMES
import time

class Game2048Player:
    def __init__(self, env):
        self.env = env

    def play(self):
        steps = 0
        achieved = 0
        while not self.env.is_game_over():
            # Get the current state of the game
            current_state = np.array(self.env.get_state())

            # Get the best move using alpha-beta pruning
            nz = np.count_nonzero(current_state)
            depth = 5 if nz > 14 else 3
            _, best_move = _alphabeta(current_state, -float('inf'), float('inf'), depth, True)
            #_, best_move = search(current_state, 3)
            achieved = max(achieved, max_tile(current_state))
            print(f"step {steps} move {ACTION_NAMES[best_move]} achieved {achieved}", end="\r")

            # Make the best move
            self.env.step(best_move)
            steps += 1
        return steps

def _alphabeta(state, alpha, beta, depth, maximizing_player):
    env = GameEnvironment(1)
    env.set_state(state)
    if depth == 0 or env.is_game_over():
        return state_score(state), None

    if maximizing_player:
        max_score = -float('inf')
        best_move = None
        for move in range(4):
            # Make a copy of the current state
            env.set_state(state)

            if env._move(move) == -1:
                # Invalid move, skip to the next move
                continue

            # Recursively call the alphabeta function with the new state
            new_score, _ = _alphabeta(env.get_state(), alpha, beta, depth - 1, False)

            # Update the maximum score and best move
            if new_score > max_score:
                max_score = new_score
                best_move = move

            # Update alpha
            alpha = max(alpha, new_score)

            # Check if beta is less than or equal to alpha
            if beta <= alpha:
                break
        return max_score, best_move
    else:
        min_score = float('inf')
        for row in range(4):
            for col in range(4):
                if state[row, col] == 0:
                    # Empty cell, can place a tile
                    for tile in [1, 2]:
                        # Make a copy of the current state
                        env.set_state(state)

                        # Place the tile on the copy
                        env.place_tile(row, col, tile)

                        # Recursively call the alphabeta function with the new state
                        new_score, _ = _alphabeta(env.get_state(), alpha, beta, depth - 1, True)

                        # Update the minimum score
                        min_score = min(min_score, new_score)

                        # Update beta
                        beta = min(beta, new_score)

                        # Check if beta is less than or equal to alpha
                        if beta <= alpha:
                            break

                    if beta <= alpha:
                        break

            if beta <= alpha:
                break

        return min_score, None

def search(state, depth):
    if (depth == 0):
        return state_score(state), None

    best = -1
    best_move = None
    env = GameEnvironment(1)
    for i in range(4):
        env.set_state(state)
        if env._move(i) == -1:
            continue
        temp = 0
        empty_slots = 0
        for row in range(4):
            for col in range(4):
                if state[row][col] == 0:
                    env.set_state(state)
                    env.place_tile(row, col, 1)
                    empty_slots += 1
                    temp += search(env.get_state(), depth - 1)[0] * 0.9
                    env.set_state(state)
                    env.place_tile(row, col, 2)
                    temp += search(env.get_state(), depth - 1)[0] * 0.1
        if empty_slots != 0:
            temp /= empty_slots
        else:
            temp = -1e+20
        if temp > best:
            best = temp
            best_move = i
    return best, best_move

        


def state_score(state):
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


def adjac_score(state):
    return adjac_score_horizon(state) + adjac_score_horizon(state.T)

def adjac_score_horizon(state):
    n = 0
    for i in range(4):
        for j in range(3):
            if state[i][j] > 0 and state[i][j+1] > 0:
              if state[i][j] == state[i][j+1]:
                n += 1
    return n


def run_with_webdriver():
    env = WebDriverGameEnvironment(0, 'file:///Users/reno/Downloads/1024game.html', False)
    player = Game2048Player(env)
    player.play()


def run():
    achieved_counts = np.zeros(20)
    sum_achieved = 0
    max_achieved = 0
    num_victory = 0

    for i in range(100):
        env = GameEnvironment(0)
        player = Game2048Player(env)
        start = time.time()
        steps = player.play()
        achieved = max_tile(env.get_state())
        achieved_counts[int(np.log2(achieved))] += 1
        sum_achieved += achieved 
        max_achieved = max(max_achieved, achieved)
        if achieved > 1024:
            num_victory += 1
        print(f"Episode {i + 1} Achieved: {max_tile(env.get_state())} in {steps} steps. Average step time: {(time.time() - start) / steps}")
    print(f"Max achieved: {max_achieved}")
    print("achievements:", achieved_counts)


if __name__ == "__main__":
    run_with_webdriver()