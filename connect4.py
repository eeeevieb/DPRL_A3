import numpy as np
import sys
import matplotlib.pyplot as plt

NUM_ROWS = 6
NUM_COLUMNS = 7
EXPLORATION_PARAMETER = 5.
EPSILON = 0.001
NUM_SIMS = 20
MAX_DEPTH = 20
MAX_CONVERGENCE_ITER = 2500

class Node:
    def __init__(self, board, last_move=None):
        self.board = board
        self.last_move = last_move  # The move that led to this node
        self.children = np.empty(NUM_COLUMNS, dtype=Node)
        self.score = 0
        self.visits = 0
        self.wins = 0

    def get_child(self, action):
        if self.children[action] is None:   # Lazy expansion
            child_board = np.copy(self.board)
            player = 3 - self.board[self.last_move] if self.last_move else 1  # opposite of last_move player, default player 1
            move = make_move(child_board, action, player)
            self.children[action] = Node(child_board, move)
        return self.children[action]

def print_tree(node, depth=0, max_depth=3):
    if depth > max_depth:  # Limit the depth to avoid excessive output
        return

    # Print the current node's details
    indent = " " * (depth * 2)
    move = node.last_move if node.last_move is not None else "Root"
    print(f"{indent}Node: {move}, Score: {node.score}, Visits: {node.visits}")

    # Recursively print children nodes
    for child in node.children:
        if child is not None:
            print_tree(child, depth + 1, max_depth)

def print_board(board, replace_old=False):
    if replace_old:                         # Replace old board printout (update it in place)
        sys.stdout.write("\033[F"*(NUM_ROWS + 1))
    print()
    for row in reversed(range(NUM_ROWS)):
        print('|', end='')
        for col in range(NUM_COLUMNS):
            token = 'X' if board[row, col] == 1 else 'O' if board[row, col] == 2 else ' '
            print(token, end='|')
        print()

# Check for 4 in a line from (row, col) in both directions along (d_row, d_col)
def check_line_match(board, row, col, dir_row, dir_col):
    player = board[row, col]
    if player == 0:
        return False

    count = 1   # The piece we're starting from from counts as 1
    # Iterate over both directions using tuple manipulation
    for dr, dc in [(dir_row, dir_col), (-dir_row, -dir_col)]:
        new_row, new_col = row + dr, col + dc
        while 0 <= new_row < NUM_ROWS and 0 <= new_col < NUM_COLUMNS and board[new_row, new_col] == player:
            count += 1
            if count == 4:
                return True
            new_row += dr
            new_col += dc

    return False

def game_state_is_terminal(board, last_move):
    if last_move is None:
        return 0  # Continue if no move has been made yet

    row, col = last_move
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Directions: horizontal, vertical, diagonal, anti-diagonal
    for dr, dc in directions:
        if check_line_match(board, row, col, dr, dc):
            return board[row, col]  # Return the winning player

    if np.all(board != 0):
        return -1  # Draw

    return 0  # Game not finished

def possible_actions(board):    # Find columns that are not yet full for the next move
    return [col for col in range(NUM_COLUMNS) if board[NUM_ROWS - 1, col] == 0]

def make_move(board, col, player):
    for row in range(NUM_ROWS):
        if board[row, col] == 0:
            board[row, col] = player
            return (row, col)

def select_best_action(node, exploration=True):     # Calculate best action based on UCB (Exploit + Explore)
    log_parent_visits = np.log(node.visits)
    best_score = -float('inf')
    best_action = None

    for action in possible_actions(node.board):
        child = node.children[action]
        if child is None and exploration:           # Prioritize exploring unvisited nodes
            return action

        # UCB formula
        if exploration:                             # Explore
            ucb_score = child.wins / child.visits
            ucb_score += EXPLORATION_PARAMETER * np.sqrt(log_parent_visits / child.visits)
        else:
            ucb_score = child.score / child.visits      # Q_Value

        if ucb_score > best_score:
            best_score = ucb_score
            best_action = action

    return best_action

def update_node_stats(node, reward):
    if reward == 1:
        node.wins += 1
    node.visits += 1
    node.score += reward

def evaluate_reward(winner):
    if winner <= 0:                         # Game not over or is a draw
        return 0
    return 1 if winner == 2 else -1         # Else +1 if player 2 wins, -1 if opponent wins

def MCTS(node, depth=MAX_DEPTH):
    winner = game_state_is_terminal(node.board, node.last_move)
    if depth == 0 or winner != 0:                   # Recursion limit or game is over
        reward = evaluate_reward(winner)            # Evaluate game state rewards
        update_node_stats(node, reward)             # Backpropagation
        return reward

    # Expansion: if we haven't explored here yet, random rollout
    if node.visits == 0:
        action = np.random.choice(possible_actions(node.board))
    # Selection: if we have explored here before, choose the best action based on UCB
    else:
        action = select_best_action(node)

    # Simulation: Recursively call MCTS
    reward = MCTS(node.get_child(action), depth - 1)

    # Backpropagation: Update the current node's reward and visits, and return the reward so ancestors can too
    update_node_stats(node, reward)
    return reward

def MCTS_until_convergence(node):
    iter = 0
    old_score = node.score / (node.visits or 1)
    q_values = []  # List to store exploit scores for plotting

    while iter < MAX_CONVERGENCE_ITER:  # In case convergence fails/takes too long
        MCTS(node)
        q_value = node.score / (node.visits or 1)
        q_values.append(q_value)  # Record the exploit score

        if 0. < abs(q_value - old_score) < EPSILON:
            break

        old_score = q_value
        iter += 1

    # Plotting the q_values
    plt.plot(q_values)
    plt.xlabel('Iteration')
    plt.ylabel('Exploit Score')
    plt.title('Exploit Score of Root Node Over Iterations')
    # plt.show()


def run_game(printout=True):
    # Board Setup
    # board = np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=int)
    # last_move = None
    # player = 1
    board = np.array([[1,2,2,2,1,1,2],
                      [0,2,1,1,2,1,0],
                      [0,2,2,2,1,2,0],
                      [0,1,1,1,2,1,0],
                      [0,2,2,1,1,2,0],
                      [0,2,1,1,2,1,0]])
    last_move = (0,0)
    player = 2
    root_node = node = Node(np.copy(board), last_move) # root_node being stored separately for printouts and debugging

    # Main Game Loop
    if printout:
        print_board(board)
    while not game_state_is_terminal(board, last_move):
        if player == 2:            # Player turn  
            MCTS_until_convergence(node)
            action = select_best_action(node, exploration=False)
        else:                   # Opponent turn   
            action = np.random.choice(possible_actions(board))

        last_move = make_move(board, action, player)
        node = node.get_child(action)
        player = 3 - player
        if printout:
            print_board(board, replace_old=True)

    print_tree(root_node)
    # Game Over & Results
    winner = game_state_is_terminal(board, last_move)
    if printout:
        print(f"Winner: {'Draw' if winner == -1 else 'Player ' + str(winner)}")
    return winner

if __name__ == '__main__':
    results = [0,0,0]
    for _ in range(NUM_SIMS):
        winner = run_game(printout=True)
        results[winner] += 1
        print(f"Results[Draw/Loss/Win]: {results}\t\t\t", end="\r")
    print()