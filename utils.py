from games.base import Game

class SelfPlayGame(object):
    """
    Wrapper for game object used during parallel self-play.

    Args:
        game: Game
            A Game object (e.g. TicTacToe, ConnectFour)
    """
    def __init__(self, game: Game) -> None:
        self.board = game.get_empty_board()
        self.memory = []

        # Initialize root node
        self.root = None
        self.node = None