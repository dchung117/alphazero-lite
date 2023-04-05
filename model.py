from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from games.base import Game

class ResNet(nn.Module):
    """
    ResNet architecture for AlphaZero

    Args:
        game: Game
            Game to be played (e.g. TicTacToe)
        n_blocks: int
            Number of layers in backbone
        n_hidden: int
            Hidden dimension size
        device: torch.device
            Device where model and batch data will be stored
    """
    def __init__(self, game: Game, n_blocks: int, n_hidden: int, device: torch.device) -> None:
        super().__init__()

        # Initial conv2d block
        self.init_block = nn.Sequential(
            nn.Conv2d(3, n_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_hidden),
            nn.ReLU()
        )

        # Define backbone
        self.back_bone = nn.ModuleList(
            [ResBlock(n_hidden) for _ in range(n_blocks)]
        )

        # Define policy and value head
        self.policy_head = nn.Sequential(
            nn.Conv2d(n_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*game.n_rows*game.n_cols, game.n_actions)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(n_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*game.n_rows*game.n_cols, 1),
            nn.Tanh()
        )

        # Load model to device
        self.device = device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ResNet model.

        param:x: torch.Tensor
            Input state tensor feature map

        return: Tuple[torch.Tensor, torch,Tensor]
            Computed policy and value of current state
        """
        # Pass through initial block and backbone
        x = self.init_block(x)
        for block in self.back_bone:
            x = block(x)

        # Compute policy logits
        policy = self.policy_head(x)

        # Compute value
        value = self.value_head(x)

        return policy, value

class ResBlock(nn.Module):
    """
    ResNet Block

    Args:
        n_hidden: int
            Hidden dimension size
    """

    def __init__(self, n_hidden: int) -> None:
        super().__init__()

        self.conv_1 = nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(n_hidden)

        self.conv_2 = nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(n_hidden)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResBlock.

        param:x: torch.Tensor
            Input tensor feature map

        return: torch.Tensor
            ResBlock output tensor feature map
        """
        # Compute residual (i.e. forward pass through both cnn blocks)
        res_x = x
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = self.bn_2(self.conv_2(x))

        # Add residual
        x += res_x

        return F.relu(x)