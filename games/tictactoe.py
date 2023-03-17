import numpy as np

class TicTacToe(object):
    """
    Python implementation of TicTacToe
    """
    def __init__(self) -> None:
        self.n_rows, self.n_cols = 3, 3
        self.n_actions = self.n_rows * self.n_cols

    def get_empty_board(self) -> np.ndarray:
        """
        Returns an empty tic-tac-toe board.

        Returns:
            np.ndarray: Empty board array
        """
        return np.zeros((self.n_rows, self.n_cols))

    def apply_move(self, board: np.array, action: int, player: int) -> np.ndarray:
        """
        Update board with move by a player.

        Args:
            board (np.ndarray): Current board array
            action (int): Action applied to board
            player (int): Player that is taking action

        Returns:
            np.ndarray: Updated board with move made
        """
        # convert action to row, col
        row = action // self.n_cols
        col = action % self.n_cols

        # update board
        board[row, col] = player

        return board

    def get_valid_moves(self, board: np.array) -> np.ndarray:
        """
        Finds empty spaces on current board (i.e. valid moves)

        Args:
            board (np.ndarray): Current board array

        Returns:
            np.ndarray: Boolean array of legal moves
        """
        return (board.flatten() == 0).astype(np.uint8)

    def check_win(self, board: np.array, action: int) -> bool:
        """
        Check if player who made recent move won game.

        Args:
            board (np.ndarray): Current board array
            action (int): Action taken by most recent player

        Returns:
            bool: Whether player won game
        """
        # return false if no action taken (i.e. root node)
        if action is None:
            return False

        # convert action to row, col
        row = action // self.n_cols
        col = action % self.n_cols

        # get player who made move
        player = board[row, col]

        # check for three in row, col, or diagonal
        win_row = board[row, :].sum() == player * self.n_cols
        win_col = board[:, col].sum() == player * self.n_rows
        win_diag = False
        if action % 2 == 0:
            center_idx = self.n_actions // 2

            # check main diagonal
            win_main_diag = False
            if abs(action - center_idx) == 4:
                win_main_diag = board.flatten()[::4].sum() == player * self.n_rows

            # check off diagonal
            win_off_diag = False
            if abs(action - center_idx) == 2:
                win_off_diag = board.flatten()[2:-2:2].sum() == player * self.n_rows

            win_diag = win_main_diag or win_off_diag

        return win_row or win_col or win_diag

    def check_end_game(self, board: np.array, action: int) -> tuple[int, bool]:
        """
        Check if game is over, return the reward (1 for win, 0 for draw) and if game ended (bool)

        Args:
            board (np.array): Current board array
            action (int): Most recent action taken

        Returns:
            tuple[int, bool]: Tuple containing reward and if game ended.
        """
        # Check if the current player won
        if self.check_win(board, action):
            return 1, True

        # Check for draw
        if self.get_valid_moves(board).sum() == 0:
            return 0, True

        # Return if game not over
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
