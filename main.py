import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)

from games.tictactoe import TicTacToe
from model import ResNet

if __name__ == "__main__":
    # Initialize game
    tictactoe = TicTacToe()
    player = 1
    board = tictactoe.get_empty_board()

    # Player 1/2 take actions
    board = tictactoe.apply_move(board, 4, 1)
    board = tictactoe.apply_move(board, 2, -1)
    print("Current board state: \n", board)
    print()

    # Encode board, save as tensor
    board = tictactoe.encode_board(board)
    print("Encoded board state: ")
    print("Player moves: \n", board[0])
    print("Opponent moves: \n", board[1])
    print("Empty moves: \n", board[2])
    print()
    state = torch.tensor(board).unsqueeze(0)

    # Initialize ResNet model, setup device
    resnet = ResNet(tictactoe, n_blocks=4, n_hidden=64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet = resnet.to(device)

    # Compute policy and value
    policy_logits, value = resnet(state.to(device))
    value = value.item()

    # Apply softmax to policy
    policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).detach().cpu().numpy()
    print("Policy: ", policy_probs)
    print("Value: ", value)

    # Plot policy
    plt.bar(range(tictactoe.n_actions), policy_probs)
    plt.title("Output policy for Player 1")
    plt.xlabel("Moves")
    plt.ylabel("Probability")
    plt.savefig("test_policy_probs.png")