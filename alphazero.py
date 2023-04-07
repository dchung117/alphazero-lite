from typing import List, Tuple, Dict, Any
import random

import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F

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
    """
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
        game: Game, kwargs: Dict[str, Any]) -> None:
        self.model = model
        self.optim = optimizer
        self.game = game
        self.kwargs = kwargs

        # Setup MCTS
        self.mcts = MCTS(game, kwargs, model)

    def self_play(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
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

            # Apply temperature-scaling to policy before sampling action
            mcts_policy_temp = mcts_policy ** (1/self.kwargs["tau"])
            mcts_policy_temp /= mcts_policy_temp.sum()

            # Sample action from MCTS policy, apply
            action = np.random.choice(self.game.n_actions, p=mcts_policy_temp)
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
                    out_memory.append((h_state, h_action, h_value))
                return out_memory

            # Change player for alpha zero to continue self play
            player = self.game.get_opponent(player)

    def train(self, memory: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        """
        Train policy and value networks on memory buffer obtained from Monte-Carlo tree search.

        Args:
            memory: List[Tuple[np.ndarray, np.ndarray, int, float]]
                Memory buffer containing tuples of board state, MCTS policy target, end-game values.
        Returns:
            None
        """
        # Shuffle training data
        random.shuffle(memory)

        # Loop through batches
        for b_idx in range(0, len(memory), self.kwargs["batch_size"]):
            # Zero-grad
            self.optim.zero_grad()

            max_b_idx = len(memory) - 1
            batch = memory[b_idx:min(b_idx+self.kwargs["batch_size"], max_b_idx)]

            # Get state, policy_targets, value_targets from memory
            state, policy_tgts, val_tgts = zip(*batch)
            state, policy_tgts, val_tgts = np.array(state), np.array(policy_tgts), np.array(val_tgts).reshape(-1, 1)
            state, policy_tgts, val_tgts = torch.tensor(state, dtype=torch.float32, device=self.model.device), \
                torch.tensor(policy_tgts, dtype=torch.float32, device=self.model.device), torch.tensor(val_tgts, dtype=torch.float32, device=self.model.device)

            # Get policy and value from model
            policy_pred, val_pred = self.model(state)

            # Compute losses
            policy_loss = F.cross_entropy(policy_pred, policy_tgts)
            value_loss = F.mse_loss(val_pred, val_tgts)
            loss = policy_loss + value_loss

            # Update model weights
            loss.backward()
            self.optim.step()

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

            # Train policy/value networks
            self.model.train()
            for _ in tqdm(range(self.kwargs["n_epochs"])):
                self.train(memory)

            # Save off checkpoint
            torch.save(self.model.state_dict(), f"model_chkpt{i}_{self.game}.pt")
            torch.save(self.optim.state_dict(), f"optim_chkpt{i}_{self.game}.pt")