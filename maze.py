import matplotlib.pyplot as plt
import numpy as np
import time

ACTIONS = [(1, 0), (-1, 0), (0, -1), (0, 1)]  # up, down, left, right
ALPHA_MIN = 0.1
EPSILON_MIN = 0.1
CALC_HYPERPARAMS = False

class MazeEnvironment:
    def __init__(self, size = 3, goal_state_rewards = None):
        self.size = size
        if goal_state_rewards is None:
            goal_state_rewards = {(size - 1, size - 1): 1.0}
        self.goal_state_rewards = goal_state_rewards
        self.state = (0,0)

    # Reset the maze to the start state
    def reset(self):
        self.state = (0,0)
        return self.state

    # Execute a step in the maze based on the given action
    def step(self, action):
        # Check if the goal is reached
        if self.state in self.goal_state_rewards:
            return self.goal_state_rewards[self.state], None    # Return the reward and indicate the episode is finished

        # Calculate the next state while ensuring it stays within maze bounds
        next_state = (max(min(self.state[0] + action[0], self.size - 1), 0),    # max(min()) to clamp values: attempting to leave maze bounds just wastes a turn
                      max(min(self.state[1] + action[1], self.size - 1), 0))

        # Update location
        self.state = next_state

        return 0, next_state        # Return a reward of 0, and the calculated next_state

    def print_maze(self):
        for i in reversed(range(self.size)):
            for j in range(self.size):
                if (i, j) == self.state:                    # Agent's current location
                    print('a', end=' ')
                elif (i, j) in self.goal_state_rewards:     # Goal state
                    print('X', end=' ')
                else:                                       # Empty cell
                    print('.', end=' ')
            print()
        print()

class TabularQLearningAgent:
    def __init__(self, maze_env, alpha_learning, alpha_decay, epsilon_initial, epsilon_decay, discount_factor):
        self.maze_env = maze_env
        self.q_table = np.zeros((maze_env.size, maze_env.size, len(ACTIONS)))
        self.epsilon_exploration = epsilon_initial
        self.alpha_learning = alpha_learning
        self.alpha_decay = alpha_decay
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor

    def learn_tabular(self, episodes):
        for _ in range(episodes):
            for i in range(self.maze_env.size):
                for j in range(self.maze_env.size):
                    for action_index, action in enumerate(ACTIONS):
                        self.maze_env.state = (i, j)  # Set current state
                        reward, next_state = self.maze_env.step(action)  # Simulate the step
                        self.update_q_table((i, j), action_index, reward, next_state)

    def learn_epsilon_greedy(self, episodes):
        for _ in range(episodes):
            self.epsilon_exploration = max(EPSILON_MIN, self.epsilon_exploration * self.epsilon_decay)
            self.alpha_learning = max(ALPHA_MIN, self.alpha_learning * self.alpha_decay)
            state = self.maze_env.reset()

            while state is not None:
                action_index = self.select_action(state)
                old_state = state
                reward, state = self.maze_env.step(ACTIONS[action_index])
                self.update_q_table(old_state, action_index, reward, state)

    def select_action(self, state):
        if np.random.rand() < self.epsilon_exploration:
            return np.random.randint(len(ACTIONS))
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def update_q_table(self, old_state, action_index, reward, state):
        if state is not None:
            reward += self.discount_factor * np.max(self.q_table[state[0], state[1]])

        self.q_table[old_state[0], old_state[1], action_index] *= (1 - self.alpha_learning)
        self.q_table[old_state[0], old_state[1], action_index] += self.alpha_learning * reward

    def print_q_table(self):
        print("Maze Q-Values:")
        for i in reversed(range(self.maze_env.size)):
            # First line for 'up' values
            for j in range(self.maze_env.size):
                up_value = "{:.2f}".format(self.q_table[i,j, 0])  # 'up' is the first action
                print(f"    {up_value}     ", end="")
            print()

            # Second line for 'left' and 'right' values
            for j in range(self.maze_env.size):
                left_value = "{:.2f}".format(self.q_table[i,j, 2])  # 'left' is the third action
                right_value = "{:.2f}".format(self.q_table[i,j, 3])  # 'right' is the fourth action
                print(f"{left_value} | {right_value} ", end=" ")
            print()

            # Third line for 'down' values
            for j in range(self.maze_env.size):
                down_value = "{:.2f}".format(self.q_table[i,j, 1])  # 'down' is the second action
                print(f"    {down_value}     ", end="")
            print("\n")
        print()

def calc_best_hyperparameters():
    # Define hyperparameter ranges
    alpha_learning_rates = [0.9, 0.7, 0.5, 0.2]
    alpha_decays = [0.9, 0.95, 0.99]
    epsilon_initials = [1.0, 0.7, 0.5]
    epsilon_decays = [0.99, 0.95, 0.90]
    discount_factors = [0.9, 0.95, 0.99]

    # Organize data by hyperparameter
    hp_dict = []
    #print('first', hp_dict)
    for lr in alpha_learning_rates:
        for lr_decay in alpha_decays:
            for eps_init in epsilon_initials:
                for eps_decay in epsilon_decays:
                    for df in discount_factors:
                        start_time = time.time()

                        for _ in range(2):  # Run each setting 10 times
                            maze = MazeEnvironment(size=3)  # Assuming a predefined MazeEnvironment
                            agent = TabularQLearningAgent(maze, lr, lr_decay, eps_init, eps_decay, df)
                            agent.learn_epsilon_greedy(episodes=350)

                        end_time = time.time()
                        total_time = end_time - start_time
                        hp_dict.append((lr, lr_decay, eps_init, eps_decay, df, total_time))

    hp_dict.sort(key=lambda x: x[-1])   # Sort by time, though note we still check correctness

    # lr, lr_decay, eps_init, eps_decay, df
    best_hyperparams = hp_dict[0][:-1]

    # for hp_set in hp_dict:
    #     print(hp_set)
    # print('best:', best_hyperparams)
    return best_hyperparams

def main():
    if CALC_HYPERPARAMS:
        hyperparameters = calc_best_hyperparameters()
    else:
        hyperparameters = (1, .99, 1, 0.999, 0.9)

    # Part 1.1: Tabular
    # print("Part 1.1 Tabular")
    # maze = MazeEnvironment(size=3)
    # agent = TabularQLearningAgent(maze, 1, 1, None, None, 0.9) 
    # agent.learn_tabular(episodes=5)
    # agent.print_q_table()

    # Part 1.2: Epsilon-Greedy
    print("Running Part 1.2 Epsilon-Greedy")
    maze = MazeEnvironment(size=3)
    agent = TabularQLearningAgent(maze, *hyperparameters) 
    agent.learn_epsilon_greedy(episodes=100)
    agent.print_q_table()

    # Part 2
    print("Running Part 2 with optimal hyperparameters...")
    goal_state_rewards = {
        (4, 4): 1.0,
        (4, 0): 0.5
    }
    maze = MazeEnvironment(size=5, goal_state_rewards=goal_state_rewards)
    maze.print_maze()
    agent = TabularQLearningAgent(maze, *hyperparameters)
    agent.learn_epsilon_greedy(episodes=1500)
    agent.print_q_table()

if __name__ == "__main__":
    main()
