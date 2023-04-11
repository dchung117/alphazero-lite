import argparse
from glob import glob
import torch
import kaggle_environments

from games.tictactoe import TicTacToe
from games.connect_four import ConnectFour
from model import ResNet

from mcts import MCTS
from agent import KaggleAgent

PLAYERS = {
    1: "User",
    -1: "Computer"
}

ENV_NAMES = {
    "TicTacToe": "tictactoe",
    "ConnectFour": "connectx"
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", type=str, choices=["TicTacToe", "ConnectFour"], help="Name of game to play.")
    parser.add_argument("--self_play", "-self", action="store_true", help="Flag to let computer play itself.")
    parser.add_argument("--chkpt", "-c", type=int, help="Checkpoint of policy/value network weights to load.")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.game == "tictactoe":
        game = TicTacToe()
        model = ResNet(game, 4, 64, device=device)
    else:
        game = ConnectFour()
        model = ResNet(game, 9, 128, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    chkpt = args.chkpt
    if not chkpt:
        chkpt = max([int(f.split("_")[1].replace("chkpt", "")) for f in glob("*.pt")]) # find most recent chkpt
    model_chkpt_file = f"model_chkpt{chkpt}_{args.game}.pt"
    optim_chkpt_file = f"optim_chkpt{chkpt}_{args.game}.pt"
    model.load_state_dict(torch.load(model_chkpt_file, map_location=device))
    optim.load_state_dict(torch.load(optim_chkpt_file, map_location=device))

    kwargs = {
        "C": 2,
        "n_searches": 600,
        "dir_epsilon": 0.0,
        "dir_alpha": 0.3,
    }
    mcts = MCTS(game, kwargs, model)

    if args.self_play:
        kwargs["dir_epsilon"] = 0.1
        kwargs["search"] = True
        kwargs["temp"] = 0

        # Create kaggle environment
        env = kaggle_environments.make(ENV_NAMES[args.game])

        player_1 = KaggleAgent(model, game, kwargs)
        player_2 = KaggleAgent(model, game, kwargs)
        players = [player_1, player_2]

        env.run(players)

        html_out = env.render(mode="html")
        with open(f"{args.game}_chkpt{chkpt}_selfplay.html", "w") as f:
            f.write(html_out)
    else:
        board = game.get_empty_board()
        player = 1

        while True:
            print("Current board: ")
            print(board)

            if player == 1: # user
                valid_moves = game.get_valid_moves(board)
                print("Valid moves: ")
                print([i for i in range(game.n_actions) if valid_moves[i]])
                action = int(input("Enter move: "))

                if valid_moves[action] == 0:
                    print("Invalid move - try again.")
                    continue
            else: # computer
                # Get neutral state
                neutral_state = game.change_perspective(board, player)
                mcts_probs = mcts.search(neutral_state)
                action = mcts_probs.argmax()

            board = game.apply_move(board, action, player)

            value, is_terminal = game.check_end_game(board, action)

            if is_terminal:
                print("GAME OVER")
                print(board)
                if value == 1:
                    print(f"Player {PLAYERS[player]} won!")
                else:
                    print("Draw")
                break

            player = game.get_opponent(player)