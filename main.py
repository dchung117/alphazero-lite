import numpy as np
import torch
torch.manual_seed(0)

from games.tictactoe import TicTacToe
from mcts import MCTS
from model import ResNet

if __name__ == "__main__":
    # Initialize game
    tictactoe = TicTacToe()
    player = 1
    board = tictactoe.get_empty_board()

    # Initialize ResNet model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet(tictactoe, 4, 64).to(device)
    model.eval()

    # Initialize MCTS
    kwargs = {
        "C": 1.4,
        "n_searches": 100
    }

    mcts = MCTS(tictactoe, kwargs, model=model, device=device)

    # Play against computer
    while True:
        print("Current board: ")
        for row in board:
            print(row)

        if player == 1:
            # Get valid moves
            valid_moves = tictactoe.get_valid_moves(board)
            print("Valid moves: ", valid_moves)
            action = int(input("Select move: "))

            # Ask again if move not valid
            if valid_moves[action] == 0:
                print("Invalid action.")
                continue
        else:
            # Use MCTS w/ policy and value networks
            neutral_state = tictactoe.change_perspective(board, player) # change board to opponent pov
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs) # select action w/ highest MCTS prob

        # Apply action
        board = tictactoe.apply_move(board, action, player)

        # Get value check if game over
        value, is_terminal = tictactoe.check_end_game(board, action)

        # Break if terminal
        if is_terminal:
            print("Game over.")
            for row in board:
                print(row)
            if value == 1:
                print(f"Player {player} won")
            else:
                print("Draw")
            break

        # Switch player
        player = tictactoe.get_opponent(player)
