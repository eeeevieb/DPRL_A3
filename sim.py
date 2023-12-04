import connect4 as game
import numpy as np

def play(iterations, depth):
    board = np.zeros((game.NUM_ROWS, game.NUM_COLUMNS), dtype=int)
    turn = 0
    while not game.game_state_is_terminal(board, None):
        if turn & 1:            # Player turn  
            root_node = game.Node(np.copy(board))    # TODO Use the same root node and simply traverse it when moves are made
            for _ in range(iterations):
                game.MCTS(root_node, turn, depth)            
            game.make_move(board, game.select_best_action(root_node), turn) #FIXME correct move based on MCTS results
        else:                   # Opponent turn   
            # choose AI move randomly among possible actions
            game.make_move(board, np.random.choice(game.possible_actions(board)), turn)
        turn += 1

        # game.display_board(board)

    winner = game.game_state_is_terminal(board, None)
    return winner
    # print(f"Winner: {'Draw' if winner == -1 else 'Player ' + str(winner)}")

if __name__ == "__main__":
    wins = 0

    for i in range(100):
        winner = play(1000, 10)
        if winner == 2:
            wins += 1
        print('iteration', i)
    
    print('Won', wins, 'percent')


"""
It 100, depth 10
71, 80

It 100, depth 100
87, 86

It 100, depth 1000
86, 86

It 1000, depth 10
90, 90, 87

It 1000, depth 100
75, 75
"""