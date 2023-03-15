from __future__ import annotations
from typing import Protocol, Optional
import numpy as np

class Game(Protocol):
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

    def change_persepective(self, board: np.ndarray, player: int) -> np.ndarray:
        ...

class Node(object):
    """
    Node (i.e. state) in Monte Carlo Tree Search algorithm
    """
    def __init__(self, game: Game, kwargs: dict, state: np.ndarray,
        parent: Optional[Node] = None, action_taken: Optional[int] = None) -> None:
        self.game = game
        self.kwargs = kwargs
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        # Initialize child node array
        self.children = []

        # Get possible moves from this state
        self.moves = game.get_valid_moves(state)

        # Initialize visit count
        self.visit_ct = 0
        self.value_sum = 0

    def is_fully_expanded(self) -> bool:
        """
        Determine if there unexplored state-action pairs.

        Returns:
            bool: Is the state action space fully expanded
        """
        return (self.moves.sum() == 0) and (len(self.children) > 0)

    def select(self) -> Node:
        """
        Select child node w/ highest upper confidence bound (UCB) score.

        Returns:
            Node: child node w/ highest UCB score
        """
        # Compute UCB scores for children
        ucb = [self.get_ucb(child) for child in self.children]

        # Select child w/ highest UCB score
        max_ucb_idx = ucb.index(max(ucb))

        return self.children[max_ucb_idx]

    def expand(self) -> Node:
        """
        Add unexlored child state of node

        Returns:
            Node: Unexplored child node
        """
        # Sample unexplored child node
        action = np.random.choice(np.where(self.moves == 1)[0], 1)

        # Create new child node
        # NOTE: expansion occurs from player 1 perspective -> game board is flipped when the new child state is created (i.e. switch to opponent's pov)
        new_state = self.game.apply_move(self.state.copy(), action, player=1)
        new_state = self.game.change_perspective(new_state, player=-1)
        new_child = Node(self.game, self.kwargs, new_state, parent=self, action_taken=action)
        self.children.append(new_child)

        # Update expandable moves - action is no longer expandable
        self.moves[action] = 0

        return new_child

    def get_ucb(self, node: Node) -> float:
        # Get Q-value of child - scale between 0 and 1
        # NOTE: child nodes are states for OPPOSITE PLAYER -> want states w/ NEGATIVE STATE
        ucb = 1 - (node.value_sum / node.visit_ct + 1)/2

        # Compute upper bound
        if self.visit_ct > 1:
            ucb += self.args["C"]*np.sqrt(self.visit_ct/node.visit_ct)

        return ucb

class MCTS(object):
    """
    Monte Carlo Tree Search (MCTS) implementation for Alpha Zero lite.
    """
    def __init__(self, game: Game, kwargs: dict) -> None:
        self.game = game
        self.kwargs = kwargs

    def search(self, state: np.ndarray):
        # Create root node
        root = Node(self.game, self.kwargs, state)

        # Run searches
        for i_search in range(self.args["n_searches"]):
            curr_node = root

            # select leaf node - find a node whose actions are not fully explored
            while curr_node.is_fully_expanded():
                # Move to child node
                curr_node = curr_node.select()

            # backpropagate if terminal node reached
            val, is_terminal = self.game.check_end_game(curr_node.state, curr_node.action)
            val = self.game.get_opponent_value()

            # expand leaf node - take an action
            # simulate until terminal node reached
            if not is_terminal:
                curr_node = curr_node.expand()
            # backpropagate result
            self.backprop()
            # expand leaf node - take an action
            # simulate until end game
            # backpropagate result