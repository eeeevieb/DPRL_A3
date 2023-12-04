import numpy as np

NUM_ROWS = 6
NUM_COLUMNS = 7
EXPLORATION_PARAMETER = np.sqrt(2.)

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

    def expand(self):
        # Expand the node by creating child nodes for each possible action
        for a in possible_actions(self.board):
            child_board = np.copy(self.board)
            move = make_move(child_board, a, self.player_to_move)
            child = Node(child_board, self, move)
            self.children[a] = child

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

def check_line_match(board, row, col, dir_row, dir_col):    #FIXME indices roll over (e.g. cylinder)
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
    #if last_move is None:
    #    return 0  # Continue if no move has been made yet
    #row, col = last_move

    for row in range(NUM_ROWS):     #FIXME Use last_move
        for col in range(NUM_COLUMNS):
            # Directions: horizontal, vertical, both diagonals
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            for dr, dc in directions:
                if check_line_match(board, row, col, dr, dc):
                    return board[row, col]  # Return the winning player

    if np.all(board != 0):
        return -1  # Draw

    return 0  # Game not finished

def possible_actions(board):
    # Find columns that are not yet full for the next move
    return [col for col in range(NUM_COLUMNS) if board[NUM_ROWS - 1, col] == 0]

def make_move(board, col, player):
    for row in range(NUM_ROWS):
        if board[row, col] == 0:
            board[row, col] = player
            return (row, col)

def evaluate_game_state(node):       # returns 1 if win, -1 if loss, 0 if game not over yet
    winner = game_state_is_terminal(node.board, node.last_move)
    if winner == 0:
        return 0
    return 1 if winner == 2 else -1

def select_best_action(node, player=2, debug=False):
    # UCB for action selection
    total_visits = sum(child.visits for child in node.children if child is not None)     #TODO is this just equal to node.visits?
    log_total_visits = np.log(total_visits)  # Avoid division by zero
    best_score = -float('inf')
    best_action = None

    for action, child in enumerate(node.children):
        if child is None:
            continue
        if child.visits == 0:
            return action  # Prioritize unvisited nodes

        # UCB formula
        exploit = child.score / child.visits
        explore = EXPLORATION_PARAMETER * np.sqrt(log_total_visits / child.visits)
        ucb_score = exploit + explore

        if debug:
            print(ucb_score)

        if player == 1:     # Invert score if evaluating for opponent's best move
            ucb_score *= -1
        if ucb_score > best_score:
            best_score = ucb_score
            best_action = action

    return best_action


def update_node_stats(node, action, score):
    # Update the node's statistics
    node.visits += 1
    node.score += score

def MCTS(node, player=2, depth=10):
    if depth == 0 or game_state_is_terminal(node.board, node.last_move):
        # Evaluate the game state and return the score
        return evaluate_game_state(node)

    # Expansion
    if not np.any(node.children):
        node.expand()
        action = np.random.choice(possible_actions(node.board))
    else:
        # Selection: Choose the best action based on UCB
        action = select_best_action(node)    

    # Simulate: Recursively call MCTS
    score = MCTS(node.children[action], player, depth - 1)

    # Backpropagation: Update the current node's statistics
    update_node_stats(node, action, score)

    return score

if __name__ == "__main__":
    # Board Setup
    board = np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=int)
    turn = 0

    # Initial Training
    node = root_node = Node(np.copy(board))
    for _ in range(1000):
        MCTS(root_node, depth=20)

    while not game_state_is_terminal(board, None):
        if turn & 1:            # Player turn  
            for _ in range(100):
                MCTS(node, depth=10)            
            action = select_best_action(node, debug=True)
            print_tree(node, max_depth=1)
        else:                   # Opponent turn   
            action = np.random.choice(possible_actions(board))
        
        make_move(board, action, 1 + (turn & 1))
        node = node.children[action]
        turn += 1

        display_board(board)

    winner = game_state_is_terminal(board, None)
    print(f"Winner: {'Draw' if winner == -1 else 'Player ' + str(winner)}")
    #print_tree(root_node, max_depth=2)