import numpy as np

NUM_ROWS = 6
NUM_COLUMNS = 7
EXPLORATION_PARAMETER = np.sqrt(2.)

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


class Node:
    def __init__(self, board, parent=None, last_move=None):
        self.board = board
        self.parent = parent
        self.last_move = last_move  # The move that led to this node
        self.children = np.empty(NUM_COLUMNS, dtype=Node)
        self.score = 0
        self.ucb = 0
        self.visits = 0

    def is_terminal(self):
        # Check if this node's state is a game-ending state
        return game_state_is_terminal(self.board, self.last_move) != 0

    def expand(self, turn):
        # Expand the node by creating child nodes for each possible action
        for a in possible_actions(self.board):
            child_board = np.copy(self.board)
            move = make_move(child_board, a, turn)
            self.children[a] = Node(child_board, self, move)

def evaluate_game_state(node):
    # Basic evaluation: return game result if terminal, otherwise return 0
    result = game_state_is_terminal(node.board, node.last_move)
    if result != 0:
        return 1 if result == 1 else -1  # Assuming AI is player 1
    return 0

def display_board(board):
    # Display the board
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

def make_move(board, col, turn):
    for row in range(NUM_ROWS):
        if board[row, col] == 0:
            board[row, col] = 1 + (turn & 1)
            return (row, col)

def evaluate_game_state(node):
    winner = game_state_is_terminal(node.board, node.last_move)
    if winner == 1:     # Opponent wins
        return -1
    elif winner == 2:   # Player wins
        return 1
    return 0            # Draw or ongoing game

def select_best_action(node):
    # Implement UCB for action selection
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

        # print(ucb_score)
        if ucb_score > best_score:
            best_score = ucb_score
            best_action = action

    return best_action


def update_node_stats(node, action, score):
    # Update the node's statistics
    # child_node = node.children[action]
    # child_node.visits += 1
    # child_node.score += score

    # Update the parent node's statistics
    node.visits += 1
    node.score += score


def MCTS(node, turn, depth=10):
    if depth == 0 or game_state_is_terminal(node.board, node.last_move):
        # Evaluate the game state and return the score
        return evaluate_game_state(node)

    # Expansion
    if not np.any(node.children):
        node.expand(turn)
        action = np.random.choice(possible_actions(node.board))
    else:
        # Selection: Choose the best action based on UCB
        action = select_best_action(node)    
    
    # Simulate: Recursively call MCTS
    child_node = node.children[action]
    score = MCTS(child_node, turn, depth - 1)

    # Backpropagation: Update the current node's statistics
    update_node_stats(node, action, score)

    return score

if __name__ == "__main__":
    board = np.array([[0,0,2,0,0,1,0],
             [0,0,2,0,0,1,0],
             [0,0,0,0,0,1,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0]])
    # display_board(board)
    root_node = Node(np.copy(board)) 
    turn = 1
    for _ in range(1000):
        MCTS(root_node, turn, depth=10)            
    #print('node:', root_node.score, root_node.visits)
    #for child in root_node.children:
    #    print(child.score, child.visits)
    print_tree(root_node)
    make_move(board, select_best_action(root_node), turn) #FIXME correct move based on MCTS results
    display_board(board)
    winner = game_state_is_terminal(board, None)
    print(f"Winner: {'Draw' if winner == -1 else 'Player ' + str(winner)}")
    
    # board = np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=int)
    # turn = 0
    # while not game_state_is_terminal(board, None):
    #     if turn & 1:            # Player turn  
    #         root_node = Node(np.copy(board))    # TODO Use the same root node and simply traverse it when moves are made
    #         for _ in range(100):
    #             MCTS(root_node, turn, depth=10)            
    #         make_move(board, select_best_action(root_node), turn) #FIXME correct move based on MCTS results
    #     else:                   # Opponent turn   
    #         # choose AI move randomly among possible actions
    #         make_move(board, np.random.choice(possible_actions(board)), turn)
    #     turn += 1

    #     display_board(board)

    # winner = game_state_is_terminal(board, None)
    # print(f"Winner: {'Draw' if winner == -1 else 'Player ' + str(winner)}")