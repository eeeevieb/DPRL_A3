import numpy as np

ALPHA_LEARNING = 0.25
EPSILON_INITIAL = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.1
GAMMA_DISCOUNT = 0.9

ACTIONS = [(1, 0), (-1, 0), (0, -1), (0, 1)]  # up, down, left, right

class Maze:
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

class Agent:
    def __init__(self, maze_env, epsilon_initial = EPSILON_INITIAL):
        self.maze_env = maze_env
        self.q_table = np.zeros((maze_env.size, maze_env.size, len(ACTIONS)))
        self.epsilon_exploration = epsilon_initial

    def learn(self, episodes):
        for _ in range(episodes):
            self.epsilon_exploration = max(EPSILON_MIN, self.epsilon_exploration * EPSILON_DECAY)
            state = self.maze_env.reset()

            while state is not None:
                action_index = self.select_action(state)
                old_state = state
                reward, state = self.maze_env.step(ACTIONS[action_index])
                self.update_q_table(old_state, action_index, reward, state)

    def select_action(self, state_index):
        if np.random.rand() < self.epsilon_exploration:
            return np.random.randint(len(ACTIONS))
        else:
            return np.argmax(self.q_table[state_index[0], state_index[1]])

    def update_q_table(self, old_state, action_index, reward, state):
        if state is not None:   # If not in terminal state, include discounted future reward
            reward += GAMMA_DISCOUNT * np.max(self.q_table[state[0], state[1]])

        # Update the Q-value of the state-action we just left
        self.q_table[old_state[0], old_state[1], action_index] *= (1 - ALPHA_LEARNING)
        self.q_table[old_state[0], old_state[1], action_index] += ALPHA_LEARNING * reward

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

if __name__ == '__main__':
    # Part 1
    maze = Maze(size = 3)
    maze.print_maze()

    agent = Agent(maze)
    agent.learn(episodes=250)
    agent.print_q_table()

    # Part 2
    goal_state_rewards = { 
                            (4, 4): 1.0,
                            (4, 0): 0.5
                            }
    maze = Maze(size = 5, goal_state_rewards = goal_state_rewards)
    maze.print_maze()

    agent = Agent(maze)
    agent.learn(episodes=2500)
    agent.print_q_table()