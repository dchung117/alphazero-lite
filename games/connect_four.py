from typing import Tuple
import numpy as np
class ConnectFour(object):
    """
    Python implementation of connect four.

    NOTE: Only top-left and top-right diagonal victories count as victories here (for simplification).
    """
    def __init__(self) -> None:
        self.n_rows = 6
        self.n_cols = 7
        self.n_actions = self.n_cols

        # in-a-row win condition
        self.n_win = 4

    def __repr__(self) -> str:
        return "ConnectFour"

    def get_empty_board(self) -> np.ndarray:
        """
        Get empty board for connect four

        Returns:
            np.ndarray: Empty board array
        """
        return np.zeros((self.n_rows, self.n_cols))

    def apply_move(self, board: np.ndarray, action: int, player: int) -> np.ndarray:
        """
        Applies move to current board

        Args:
            board (np.ndarray): Current board
            action (int): Column that player will place piece
            player (int): Player ID

        Returns:
            np.ndarray: Updated board after move applied
        """
        # Identify bottom-most row that is unfilled
        bottom_row = np.argwhere(board[:, action] == 0).max()

        # Update board
        board[bottom_row, action] = player

        return board

    def get_valid_moves(self, board: np.ndarray) -> np.ndarray:
        """
        Get all valid moves for current board

        Args:
            board (np.ndarray): Current board array

        Returns:
            np.ndarray: Boolean array of legal moves
        """
        # Check first row for any unfilled pieces
        return (board[0] == 0).astype(np.uint8)

    def check_win(self, board: np.ndarray, action: int) -> bool:
        """
        Checking if most recent move played resulted in win

        Args:
            board (np.ndarray): Current board array
            action (int): Action taken by most recent player

        Returns:
            bool: Whether player won game
        """
        # Return false for no action
        if action is None:
            return False

        # Get top-most filled row
        row = np.argwhere(board[:, action] != 0).min()
        column = action

        # Get current player
        player = board[row][column]

        def count(offset_row: int, offset_column: int) -> int:
            """
            Helper function to walk along row/col/diagonal and count consecutive player pieces

            Args:
                offset_row (int): Row direction to move along (left or right)
                offset_column (int): Column direction to move along (up or down)

            Returns:
                int: Number of consecutive player pieces after walk
            """
            for i in range(1, self.n_win):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0 
                    or r >= self.n_rows
                    or c < 0 
                    or c >= self.n_cols
                    or board[r][c] != player
                ):
                    return i - 1
            return self.n_win - 1

        return (
            count(1, 0) >= self.n_rows - 1 # column -> check if there are at least 3 consecutive player pieces below
            or (count(0, 1) + count(0, -1)) >= self.n_win - 1 # row -> check if there are at least 3 consecutive pieces left or right
            or (count(1, 1) + count(-1, -1)) >= self.n_win - 1 # left diagonal -> check if there are at least 3 consecutive pieces moving up/down left-diagonal
            or (count(1, -1) + count(-1, 1)) >= self.n_win - 1 # right diagonal -> check if there are at least 3 consecutive pieces moving up/down right-diagonal
        )

    def check_end_game(self, board: np.ndarray, action: int) -> Tuple[int, bool]:
        """
        Check if game is over, return the reward (1 for win, 0 for draw) and if game ended (bool)

        Args:
            board (np.array): Current board array
            action (int): Most recent action taken

        Returns:
            tuple[int, bool]: Tuple containing reward and if game ended.
        """
        # Check if current player won
        if self.check_win(board, action):
            return 1, True

        # Check if tie
        if self.get_valid_moves(board).sum() == 0:
            return 0, True

        # Otherwise, game not over
        return 0, False

    def get_opponent(self, player: int) -> int:
        """
        Get opponent player ID

        Args:
            player (int): Current layer ID

        Returns:
            int: Opponent player ID
        """
        return -player

    def get_opponent_value(self, value: int) -> int:
        """
        Negate the value of opposing player

        Args:
            value (int): Value of opposing player that won

        Returns:
            int: Negated value of opposing player
        """
        return -value

    def change_perspective(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        Alter current board array point of view to the opponent's.

        Args:
            board (np.ndarray): Current board array
            player (int): Opponent player ID

        Returns:
            np.ndarray: Board array w/ flipped player perspective
        """
        return board*player

    def encode_board(self, board: np.ndarray) -> np.ndarray:
        """
        Encode board into 3-channel array (i.e. player moves, opponent moves, vacant spaces).

        Args:
            board (np.ndarray): Current board array

        Returns:
            np.ndarray: Encoded board array represented as 3-channel array
        """
        # Get player/opp moves and vacant spaces
        player_moves = (board == 1)
        opp_moves = (board == -1)
        empty_spaces = (board == 0)

        # Stack arrays
        enc_board = np.stack((player_moves, opp_moves, empty_spaces), axis=0).astype(np.float32)

        # Swap batch and channel axes (if multiple boards passed)
        if len(board.shape) == 3:
            enc_board = enc_board.swapaxes(0, 1)

        return enc_board