from __future__ import annotations
from typing import Protocol, Optional
import numpy as np
from games.tictactoe import TicTacToe

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

    def change_perspective(self, board: np.ndarray, player: int) -> np.ndarray:
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

    def simulate(self) -> float:
        """
        Simulate a game played from current node.

        Returns:
            float: Value from current player perspective after simulating rollout from current node
        """
        # Get node value and check if game is terminated
        value, is_terminal = self.game.check_end_game(self.state, self.action_taken)

        # Change perspective of value (i.e. other player)
        value = self.game.get_opponent(value)

        # Return if game is over
        if is_terminal:
            return value

        # Rollout game until over
        rollout_state = self.state.copy()
        rollout_player = 1
        while True:
            # Select random action from available moves
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])

            # Apply action to get next state
            rollout_state = self.game.apply_move(rollout_state, action, rollout_player)

            # Check value
            value, is_terminal = self.game.check_end_game(rollout_state, action)

            # Return value if game over
            if is_terminal:
                # Reverse value if opponent made move
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value

            # Change players
            rollout_player = self.game.get_opponent(rollout_player)

    def backprop(self, value: float) -> None:
        """
        Updating nodes from child to parent based on outcome of simulation.

        Args:
            value (float): Value of simulation from current player perspective.
        """
        # Update value sum and visit ct
        self.value_sum += value
        self.visit_ct += 1

        # Pass value to parent (state from opponent perspective)
        if self.parent != None:
            value = self.game.get_opponent_value(value)
            self.parent.backprop(value)

    def get_ucb(self, node: Node) -> float:
        # Get Q-value of child - scale between 0 and 1
        # NOTE: child nodes are states for OPPOSITE PLAYER -> want states w/ NEGATIVE STATE
        ucb = 1 - (node.value_sum / node.visit_ct + 1)/2

        # Compute upper bound
        if self.visit_ct > 1:
            ucb += self.kwargs["C"]*np.sqrt(self.visit_ct/node.visit_ct)

        return ucb

class MCTS(object):
    """
    Monte Carlo Tree Search (MCTS) implementation for Alpha Zero lite.
    """
    def __init__(self, game: Game, kwargs: dict) -> None:
        self.game = game
        self.kwargs = kwargs

    def search(self, state: np.ndarray) -> np.ndarray:
        """
        Apply MCTS starting from a specified game state.

        Args:
            state (np.ndarray): Array representation of current game state

        Returns:
            np.ndarray: Policy for current state after MCTS
        """
        # Create root node
        root = Node(self.game, self.kwargs, state)

        # Run searches
        for i_search in range(self.kwargs["n_searches"]):
            curr_node = root

            # select leaf node - find a node whose actions are not fully explored
            while curr_node.is_fully_expanded():
                # Move to child node
                curr_node = curr_node.select()

            # backpropagate if terminal node reached
            val, is_terminal = self.game.check_end_game(curr_node.state, curr_node.action_taken)
            val = self.game.get_opponent_value(val)

            # expand leaf node - take an action
            # simulate until terminal node reached
            if not is_terminal:
                curr_node = curr_node.expand()
                val = curr_node.simulate()

            # backpropagate node value up tree
            curr_node.backprop(val)

        # Return distribution of visit cts from root node
        action_probs = np.zeros(self.game.n_actions)
        for child in root.children:
            # Get visit cts of actions from root (i.e. backpropagated from many games rolled out)
            action_probs[child.action_taken] = child.visit_ct
        action_probs /= action_probs.sum()

        return action_probs