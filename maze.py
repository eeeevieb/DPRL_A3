import numpy as np

MAZE_SIZE = 5
ALPHA_LEARNING = 0.2
EPSILON_EXPLORATION = 0.4
GAMMA_DISCOUNT = 0.9

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

class Maze:
    def __init__(self, start=(0, 0), goal=(4, 4), reward=1.0):
        self.start = start
        self.goal = goal
        self.reward = reward
        self.state = start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        # Calculate the next state
        next_state = (max(min(self.state[0] + action[0], MAZE_SIZE - 1), 0),
                      max(min(self.state[1] + action[1], MAZE_SIZE - 1), 0))

        # Check if the goal is reached
        done = (next_state == self.goal)
        reward = self.reward if done else 0

        self.state = next_state
        return next_state, reward, done

    def print_maze(self):
        for i in range(MAZE_SIZE):
            for j in range(MAZE_SIZE):
                if (i, j) == self.state:
                    print('o', end=' ')
                elif (i, j) == self.goal:
                    print('X', end=' ')
                else:
                    print('.', end=' ')
            print()

class TabularQLearningAgent:
    def __init__(self, maze_env):
        self.maze_env = maze_env
        self.q_table = np.zeros((MAZE_SIZE, MAZE_SIZE, len(ACTIONS)))

    def learn(self, episodes=10000):
        for _ in range(episodes):
            state = self.maze_env.reset()
            done = False

            while not done:
                action_index = self.select_action(state)
                next_state, reward, done = self.maze_env.step(ACTIONS[action_index])
                self.update_q_table(state, action_index, reward, next_state)
                state = next_state

    def select_action(self, state_index):
        if np.random.rand() < EPSILON_EXPLORATION:
            return np.random.randint(len(ACTIONS))
        else:
            return np.argmax(self.q_table[state_index[0], state_index[1]])

    def update_q_table(self, state, action_index, reward, next_state):
        next_max = np.max(self.q_table[next_state[0], next_state[1]])

        # Calculate new Q-value
        self.q_table[state[0], state[1], action_index] *= (1 - ALPHA_LEARNING)
        self.q_table[state[0], state[1], action_index] += ALPHA_LEARNING * (reward + GAMMA_DISCOUNT * next_max)

    def print_q_values(self):
        print("Maze Q-Values:")
        for i in range(MAZE_SIZE):
            # First line for 'up' values
            for j in range(MAZE_SIZE):
                up_value = "{:.2f}".format(self.q_table[i,j, 0])  # Assuming 'up' is the first action
                print(f"    {up_value}     ", end="")
            print()

            # Second line for 'left' and 'right' values
            for j in range(MAZE_SIZE):
                left_value = "{:.2f}".format(self.q_table[i,j, 2])  # Assuming 'left' is the third action
                right_value = "{:.2f}".format(self.q_table[i,j, 3])  # Assuming 'right' is the fourth action
                print(f"{left_value} | {right_value} ", end=" ")
            print()

            # Third line for 'down' values
            for j in range(MAZE_SIZE):
                down_value = "{:.2f}".format(self.q_table[i,j, 1])  # Assuming 'down' is the second action
                print(f"    {down_value}     ", end="")
            print("\n")  # Extra newline for spacing between rows

    def print_q_table(self):
        print(f"                 up     down    left    right")
        for row in range(MAZE_SIZE):
            for col in range(MAZE_SIZE):
                q_values_formatted = ["{:.2f}".format(q) for q in self.q_table[row, col]]
                print(f"State ({row}, {col}): {q_values_formatted}")

maze = Maze()
agent = TabularQLearningAgent(maze)
agent.learn()

maze.reset()
maze.print_maze()
agent.print_q_values()