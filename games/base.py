from typing import Protocol
import numpy as np

class Game(Protocol):
    @property
    def n_rows(self) -> int:
        ...

    @property
    def n_cols(self) -> int:
        ...

    @property
    def n_actions(self) -> int:
        ...

    def get_empty_board(self) -> None:
        ...

    def apply_move(self, board: np.array, action: int, player: int) -> np.ndarray:
        ...

    def get_valid_moves(self, board: np.ndarray) -> np.ndarray:
        ...

    def check_win(self, board: np.ndarray, action: int) -> bool:
        ...

    def check_end_game(self, board: np.ndarray, action: int) -> tuple[int, bool]:
        ...

    def get_opponent(self, player: int) -> int:
        ...

    def get_opponent_value(self, value: int) -> int:
        ...

    def change_perspective(self, board: np.ndarray, player: int) -> np.ndarray:
        ...

    def encode_board(self, state: np.ndarray) -> np.ndarray:
        ...