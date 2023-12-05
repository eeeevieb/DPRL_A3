import numpy as np
import time

NUM_ROWS = 6
NUM_COLUMNS = 7
EXPLORATION_PARAMETER = np.sqrt(2.)
EPSILON = 0.01
NUM_SIMS = 100

class Node:
    def __init__(self, board, parent=None, last_move=None):
        self.board = board
        self.parent = parent
        self.last_move = last_move  # The move that led to this node
        self.player_to_move = 1 if last_move==None else 3 - (board[last_move])  # opposite of prev player
        self.children = np.empty(NUM_COLUMNS, dtype=Node)
        self.score = 0
        self.ucb = 0
        self.visits = 0

    def get_child(self, action):
        if self.children[action] is None:   # Lazy expansion
            child_board = np.copy(self.board)
            move = make_move(child_board, action, self.player_to_move)
            self.children[action] = Node(child_board, self, move)
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

def display_board(board):
    print()
    for row in reversed(range(NUM_ROWS)):
        print('|', end='')
        for col in range(NUM_COLUMNS):
            token = 'X' if board[row, col] == 1 else 'O' if board[row, col] == 2 else ' '
            print(token, end='|')
        print()

def check_line_match(board, row, col, dir_row, dir_col):
    """Check for 4 in a line from (row, col) in both directions along (d_row, d_col)"""
    player = board[row, col]
    if player == 0:
        return False

    count = 1  # The piece we're starting from from counts as 1
    for dr, dc in [(dir_row, dir_col), (-dir_row, -dir_col)]:   # Iterate over both directions using tuple manipulation
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

def select_best_action_ucb(node, player=2):
    # UCB for action selection
    log_parent_visits = np.log(node.visits or 1)  # Avoid division by zero
    best_score = -float('inf')
    best_action = None

    for action, child in enumerate(node.children):
        if child is None:                           #TODO what to do with unvisited children?
            return action

        # UCB formula
        exploit = child.score / child.visits
        explore = EXPLORATION_PARAMETER * np.sqrt(log_parent_visits / child.visits)
        ucb_score = exploit + explore

        #if player == 1:     # Invert score if evaluating for opponent's best move; currently unused since we aren't using maximin and opponent is random
        #    ucb_score *= -1
        if ucb_score > best_score:
            best_score = ucb_score
            best_action = action

    return best_action

def update_node_stats(node, reward):
    node.visits += 1
    node.score += reward

def evaluate_reward(winner, player):
    if winner <= 0:
        return 0
    return 1 if winner == player else -1

def MCTS(node, depth=20, player=2):
    winner = game_state_is_terminal(node.board, node.last_move)
    if depth == 0 or winner != 0:                   # Recursion limit or game is over
        reward = evaluate_reward(winner, player)    # Evaluate game state rewards
        update_node_stats(node, reward)             # Backpropagation
        return reward

    # Expansion: if we haven't explored here yet: random rollout
    if node.visits == 0:
        action = np.random.choice(possible_actions(node.board))
    # Selection: if we have explored here before: choose the best action based on UCB
    else:
        action = select_best_action_ucb(node)

    # Simulation: Recursively call MCTS
    reward = MCTS(node.get_child(action), depth - 1, player)    # other player's turn, so rewards for them are negative for us

    # Backpropagation: Update the current node's reward and visits, and return the reward so ancestors can too
    update_node_stats(node, reward)
    return reward

def MCTS_until_convergence(node, max_iter=2500):  # Run MCTS repeatedly until exploit score stops changing much 
    iter = 0
    old_score = node.score / ( node.visits or 1 )
    while iter < max_iter:      # In case convergence fails/takes too long
        MCTS(node)
        exploit_score = node.score / ( node.visits or 1 )

        if 0. < abs(exploit_score - old_score) < EPSILON:
            return

        old_score = exploit_score
        iter += 1

def run_game(printout=True):
    # Board Setup
    board = np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=int)
    last_move = None
    turn = 0
    root_node = node = Node(np.copy(board)) # root_node being stored separately for printouts and debugging

    # Main Game Loop
    while not game_state_is_terminal(board, last_move):
        if turn & 1:            # Player turn  
            MCTS_until_convergence(node)
            action = select_best_action_ucb(node)       # TODO: Set exploration parameter to 0 when actually picking real move?
        else:                   # Opponent turn   
            action = np.random.choice(possible_actions(board))

        last_move = make_move(board, action, 1 + (turn & 1))
        node = node.get_child(action)
        turn += 1
        if printout:
            display_board(board)

    # Game Over & Results
    winner = game_state_is_terminal(board, last_move)
    if printout:
        print(f"Winner: {'Draw' if winner == -1 else 'Player ' + str(winner)}")
    return winner

if __name__ == '__main__':
    #start = time.time()
    results = [0,0,0]
    for _ in range(NUM_SIMS):
        winner = run_game(printout=False)
        results[winner] += 1
    #print("Time:", (time.time() - start))
    print(results)