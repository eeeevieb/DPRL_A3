import numpy as np
import cProfile

MAZE_SIZE = 5
ALPHA_LEARNING = 0.1
EPSILON_CONVERGENCE = 0.1
GAMMA_DISCOUNT = 0.9

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

class MazeEnvironment:
    def __init__(self, start=(0, 0), goal=(4, 4), reward=1.0):
        self.start = start
        self.goal = goal
        self.reward = reward
        self.state = start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, delta):
        # Calculate the next state
        next_state = (max(min(self.state[0] + delta[0], MAZE_SIZE - 1), 0),
                      max(min(self.state[1] + delta[1], MAZE_SIZE - 1), 0))

        # Check if the goal is reached
        done = (next_state == self.goal)
        reward = self.reward if done else 0

        self.state = next_state
        return next_state, reward, done

    def print_maze(self):
        for i in range(MAZE_SIZE):
            for j in range(MAZE_SIZE):
                if (i, j) == self.state:
                    print('@', end=' ')
                elif (i, j) == self.goal:
                    print('X', end=' ')
                else:
                    print('.', end=' ')
            print()

class TabularQLearningAgent:
    def __init__(self, maze_env):
        self.maze_env = maze_env
        self.q_table = np.zeros((MAZE_SIZE**2, len(ACTIONS)))

    def learn(self, episodes=100):
        for _ in range(episodes):
            state = self.maze_env.reset()
            done = False

            while not done:
                current_state_index = self.get_state_index(state)
                action_index = self.select_action(current_state_index)
                next_state, reward, done = self.maze_env.step(ACTIONS[action_index])
                self.update_q_table(current_state_index, action_index, reward, next_state)
                state = next_state

    def get_state_index(self, state):
        return state[0] * MAZE_SIZE + state[1]

    def select_action(self, state_index):
        if np.random.rand() < EPSILON_CONVERGENCE:
            return np.random.randint(len(ACTIONS))
        else:
            return np.argmax(self.q_table[state_index])

    def update_q_table(self, state, action, reward, next_state):
        next_state_index = self.get_state_index(next_state)

        # Get maximum Q-value for the next state
        next_max = np.max(self.q_table[next_state_index])

        # Update Q-value
        self.q_table[state, action] *= (1 - ALPHA_LEARNING)
        self.q_table[state, action] += ALPHA_LEARNING * (reward + GAMMA_DISCOUNT * next_max)

def main():
    maze_env = MazeEnvironment()
    agent = TabularQLearningAgent(maze_env)
    agent.learn()

    # Inspect the learned Q-table
    print(agent.q_table)

cProfile.run('main()', 'profile_stats')
import pstats
p = pstats.Stats('profile_stats')
p.sort_stats('cumulative').print_stats(10)
# # Example usage
# maze_env = MazeEnvironment()
# print("Initial state:", maze_env.reset())
# maze_env.print_maze()

# next_state, reward, done = maze_env.step('right')
# print("Next state:", next_state, "Reward:", reward, "Done:", done)
# maze_env.print_maze()