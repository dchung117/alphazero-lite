import numpy as np

class TicTacToe(object):
    """
    Python implementation of TicTacToe

    Args:
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

if __name__ == "__main__":
    # Initialize game
    tictactoe = TicTacToe()
    player = 1
    board = tictactoe.get_empty_board()
    reward, is_over = 0, False

    # Continue playing moves until end game
    while True:
        # Get next set of valid moves
        valid_moves = tictactoe.get_valid_moves(board)

        # Take action
        action = np.random.choice(tictactoe.n_actions, p=valid_moves/valid_moves.sum())
        board = tictactoe.apply_move(board, action, player)

        # Check if game is over
        reward, is_over = tictactoe.check_end_game(board, action)

        # Print winning player if game is over
        if is_over:
            break
        else: # print current board state
            print("New board: ")
            for row in board:
                print(row.tolist())

        # Change current player
        player = tictactoe.get_opponent(player)

    # Print end game outcome
    if reward == 1:
        print(f"Player {player} won!")
    else:
        print("Draw.")
    print(f"End config: ")
    for row in board:
        print(row.tolist())
