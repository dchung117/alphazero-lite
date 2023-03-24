import numpy as np
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
        "n_self_plays": 10,
        "n_epochs": 4
    }

    # Create AlphaZeroLite object
    alpha_zero = AlphaZeroLite(model, optim, tictactoe, kwargs, device=device)
    alpha_zero.learn()