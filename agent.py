
import numpy as np
import torch
from torch import nn

from games.base import Game
from mcts import MCTS

class KaggleAgent(object):
    """
    Kaggle agent class for self-play in Kaggle Environments

    Args:
        model: nn.Module
            Policy and value network
        game: Game
            Game to be played (i.e. TicTacToe or ConnectFour)
        kwargs: dict
            Keyword arguments used in AlphaZeroLite
    """
    def __init__(self, model: nn.Module, game: Game, kwargs: dict) -> None:
        self.model = model
        self.model.eval()
        self.game = game
        self.kwargs = kwargs

        self.mcts = None
        if self.kwargs["search"]:
            self.mcts = MCTS(game, kwargs, model)

    def run(self, obs: dict, conf: dict) -> int:
        """
        Method for agent to take an action.

        Args:
            obs: dict
                Dictionary of observations
                    mark - current player (1 or -1)
                    board - board state
            conf: dict
                Configuration dictionary

        Returns: int
            Action taken by player
        """
        player = obs["mark"] if obs["mark"] == 1 else -1
        board = np.array(obs["board"].reshape(self.game.n_rows, self.game.n_cols))
        board[board == 2] = -1

        board = self.game.change_perspective(board, player)
        if self.mcts:
            policy = self.mcts.search(board)
        else:
            with torch.inference_mode():
                policy, _ = self.model(torch.tensor(self.game.encode_board(board)))

        valid_moves = self.game.get_valid_moves(board)
        policy *= valid_moves
        policy /= policy.sum()

        if self.kwargs["temp"] == 0:
            action = int(policy.argmax())
        elif self.kwargs["temp"] == float("inf"):
            action = np.random.choice([i for i in range(self.game.n_actions) if policy[i] > 0])
        else:
            policy = policy ** (1/self.kwargs["temp"])
            policy /= policy.sum()
            action = np.random.choice([i for i in range(self.game.n_actions)], p=policy)

        return action