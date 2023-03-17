import numpy as np

from games.tictactoe import TicTacToe
from mcts import MCTS

if __name__ == "__main__":
    # Initialize game
    tictactoe = TicTacToe()
    player = 1
    board = tictactoe.get_empty_board()
    reward, is_over = 0, False

    kwargs = {"C": np.sqrt(2), "n_searches": 1000}
    mcts = MCTS(tictactoe, kwargs)

    # Continue playing moves until end game
    while True:
        if player == 1: # take action if current player
            # Get next set of valid moves
            valid_moves = tictactoe.get_valid_moves(board)
            action_idxs = [i for i,v in enumerate(valid_moves) if v == 1]
            print("Valid actions: ", action_idxs)

            # Ask for move input
            action = int(input(f"Enter move from {action_idxs}: "))


        else: # opponent will use MCTS to take action
            # Change to opponent perspective (i.e. self-play)
            neutral_state = tictactoe.change_perspective(board, player)

            # Get action probabilities from MCTS
            action_probs = mcts.search(neutral_state)

            # Take most probable action
            action = action_probs.argmax()

        # Apply move (either user or computer)
        board = tictactoe.apply_move(board, action, player)

        # Check if game is over
        reward, is_over = tictactoe.check_end_game(board, action)

        # Print winning player if game is over
        if is_over:
            break
        else: # print current board state
            print("New board: ")
            for row in board:
                print(row.tolist())

        # Change current player
        player = tictactoe.get_opponent(player)

    # Print end game outcome
    if reward == 1:
        print(f"Player {player} won!")
    else:
        print("Draw.")
    print(f"End config: ")
    for row in board:
        print(row.tolist())