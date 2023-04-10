import numpy as np
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
from torch.optim import Adam

from games.connect_four import ConnectFour
from model import ResNet
from alphazero import AlphaZeroLite, AlphaZeroLiteParallel

if __name__ == "__main__":
    # Initialize game
    game = ConnectFour()
    player = 1
    board = game.get_empty_board()

    # Initialize ResNet model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, 9, 128, device=device)
    model.eval()

    # Initialize adam optimizer
    optim = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # Initialize AlphaZero hyperparameters
    kwargs = {
        "C": 2,
        "n_searches": 600,
        "n_iters": 8,
        "n_self_plays": 500,
        "n_parallel": 100,
        "n_epochs": 4,
        "batch_size": 128,
        "tau": 1.25, # policy temperature for action sampling
        "dir_epsilon": 0.25, # Dirichlet epsilon for pre-MCTS node expansion/exploration of root
        "dir_alpha": 0.3 # Dirichlet alpha
    }

    # Create AlphaZeroLite object
    alpha_zero = AlphaZeroLiteParallel(model, optim, game, kwargs)
    alpha_zero.learn()

    # Test out trained model
    board = game.apply_move(board, 0, 1)
    board = game.apply_move(board, 1, 1)
    board = game.apply_move(board, 2, 1)
    board = game.apply_move(board, 6, -1)
    board = game.apply_move(board, 5, -1)

    encoded_state = game.encode_board(board)

    tensor_state = torch.tensor(encoded_state, device=model.device).unsqueeze(0)

    policy, value = alpha_zero.model(tensor_state)
    value = value.cpu().item()
    policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

    print("Value: ", value)

    print("Board state: ", board)
    print("Encoded state: ", tensor_state)

    plt.bar(range(game.n_actions), policy)
    plt.savefig(f"alphazero_policy_chkpt_{game}.png")
