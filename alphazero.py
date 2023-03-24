from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from tqdm import tqdm

from games.base import Game
from mcts import MCTS

class AlphaZeroLite(object):
    """
    AlphaZero lite implementation.

    Args:
        model: torch.nn.Module
            Policy and value networks
        optimizer: torch.optim.Optimizer
            Torch optimizer for training
        game: Game
            Game that agent will learn to play
        kwargs: Dict[str, Any]
            Lookup table of hyperparameters
        device: torch.device
            Device where model and batch data will be stored
    """
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
        game: Game, kwargs: Dict[str, Any], device: torch.device) -> None:
        self.model = model
        self.optim = optimizer
        self.game = game
        self.kwargs = kwargs
        self.device = device

        # Setup MCTS
        self.mcts = MCTS(game, kwargs, model, device)

    def self_play(self) -> List[Tuple[np.ndarray, np.ndarray, int, float]]:
        """
        Agent plays itself and collects game data w/ outcomes via Monte-Carlo tree search.

        Args:
            None
        Returns: List[Tuple[np.ndarray, np.ndarray, int, float]]
            memory buffer containing board states, MCTS policies, current player, value
        """
        # Initialize memory, player, game
        memory = []
        player = 1
        state = self.game.get_empty_board()

        # Computer plays oneself via Monte Carlo tree search
        while True:
            # Sample action probabilities via MCTS
            neutral_state = self.game.change_perspective(state, player)
            mcts_policy = self.mcts.search(neutral_state)

            # Append neutral state, mcts_policy, player to memory
            # NOTE: state always from perspective of CURRENT PLAYER
            memory.append((neutral_state, mcts_policy, player))

            # Sample action from MCTS policy, apply
            action = np.random.choice(self.game.n_actions, p=mcts_policy)
            state = self.game.apply_move(state, action, player)

            # Check if game is over
            value, is_terminal = self.game.check_end_game(state, action)
            if is_terminal: # Append value to memory buffer
                out_memory = []
                for h_state, h_action, h_player in memory:
                    # Value should reflect player that moved in current turn
                    h_value = value if h_player == player else self.game.get_opponent_value(value)

                    # Encode state as 3-channel tensor
                    h_state = self.game.encode_board(h_state)
                    out_memory.append((h_state, h_action, h_player, h_value))
                return out_memory

            # Change player for alpha zero to continue self play
            player = self.game.get_opponent(player)

    def train(self, memory: list):
        pass

    def learn(self):
        """
        Function that executes data collection via self-play and policy/value network training.

        Keyword arguments:
        argument None
        Return: None
        """
        # Run learning iterations
        for i in tqdm(range(self.kwargs["n_iters"])):
            # Initialize memory buffer
            memory = []

            # Loop over self-play games
            self.model.eval()
            for _ in tqdm(range(self.kwargs["n_self_plays"])):
                # Collect memory from self-play
                memory += self.self_play()

            # Train polic/value networks
            for _ in tqdm(range(self.kwargs["n_epochs"])):
                self.train(memory)

            # Save off checkpoint
            torch.save(self.model.state_dict(), f"model_chkpt{i}.pt")
            torch.save(self.optim.state_dict(), f"optim_chkpt{i}.pt")