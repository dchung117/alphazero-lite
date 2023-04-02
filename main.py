import numpy as np
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
from torch.optim import Adam

from games.tictactoe import TicTacToe
from model import ResNet
from alphazero import AlphaZeroLite

if __name__ == "__main__":
    # Initialize game
    tictactoe = TicTacToe()
    player = 1
    board = tictactoe.get_empty_board()

    # Initialize ResNet model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet(tictactoe, 4, 64).to(device)
    model.eval()

    # Initialize adam optimizer
    optim = Adam(model.parameters(), lr=0.001)

    # Initialize AlphaZero hyperparameters
    kwargs = {
        "C": 1.4,
        "n_searches": 60,
        "n_iters": 3,
        "n_self_plays": 500,
        "n_epochs": 4,
        "batch_size": 64
    }

    # Create AlphaZeroLite object
    alpha_zero = AlphaZeroLite(model, optim, tictactoe, kwargs, device=device)
    alpha_zero.learn()

    # Test out trained model
    board = tictactoe.apply_move(board, 2, -1)
    board = tictactoe.apply_move(board, 4, -1)
    board = tictactoe.apply_move(board, 6, 1)
    board = tictactoe.apply_move(board, 8, 1)

    encoded_state = tictactoe.encode_board(board)

    tensor_state = torch.tensor(encoded_state).unsqueeze(0)

    policy, value = alpha_zero.model(tensor_state.to(device))
    value = value.cpu().item()
    policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

    print("Value: ", value)

    print("Board state: ", board)
    print("Encoded state: ", tensor_state)

    plt.bar(range(tictactoe.n_actions), policy)
    plt.savefig("alphazero_policy_chkpt.png")