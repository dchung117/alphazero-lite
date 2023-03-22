from __future__ import annotations
from typing import Optional
import numpy as np
import torch
from torch import nn

from games.base import Game

class Node(object):
    """
    Node (i.e. state) in Monte Carlo Tree Search algorithm

    Args:
        game: Game
            Game to be played (e.g. TicTacToe)
        kwargs: dict
            Keyword arguments containing hyperparameters
        state: np.ndarray
            Game state array of node
        parent: Optional[Node] = None
            Parent node of current state
        action_taken: Optional[int] = None
            Action taken to reach this state
        prior: int = 0
            Prior policy probability of reaching this state
    """
    def __init__(self, game: Game, kwargs: dict, state: np.ndarray,
        parent: Optional[Node] = None, action_taken: Optional[int] = None, prior: int = 0, is_alpha: bool = False) -> None:
        self.game = game
        self.kwargs = kwargs
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.is_alpha = is_alpha

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

    def expand(self, policy: Optional[np.ndarray] = None) -> Node:
        """
        Add unexlored child state of node

        Args:
            policy (Optional[np.ndarray]) = None: Policy used to select actions
        Returns:
            Optional[Node]: Unexplored child node (if no policy given), else returns None
        """
        # Expand each child node if given policy
        if policy is not None:
            for action, prob in enumerate(policy):
                # Expand every possible action
                if prob > 0:
                    new_state = self.game.apply_move(self.state.copy(), action, player=1)
                    new_state = self.game.change_perspective(new_state, player=-1)
                    new_child = Node(self.game, self.kwargs, new_state, parent=self, action_taken=action, prior=prob, is_alpha=True)
                    self.children.append(new_child)

                self.moves[action] = 0
        else:
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
            # Select action from policy
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
        """
        Compute upper confidence bound of child node during selection.

        Args:
            node: Node
                Child node in MCTS
        """
        # Get Q-value of child - scale between 0 and 1
        # NOTE: child nodes are states for OPPOSITE PLAYER -> want states w/ NEGATIVE STATE
        if self.is_alpha and (node.visit_ct) == 0:
            q_val = 0
        else:
            q_val = 1 - ((node.value_sum / node.visit_ct) + 1)/2

        # Compute upper limit
        if self.is_alpha: # MCTS w/ policy network
            upper_limit = self.kwargs["C"]*np.sqrt(self.visit_ct/(node.visit_ct + 1))*self.prior
        else: # standard MCTS
            upper_limit = self.kwargs["C"]*np.sqrt(np.log(self.visit_ct)/node.visit_ct)
        ucb = q_val + upper_limit

        return ucb

class MCTS(object):
    """
    Monte Carlo Tree Search (MCTS) implementation for Alpha Zero lite.

    Args:
        game: Game
            Game to be played (e.g. TicTacToe)
        kwargs: dict
            Keyword arguments containing hyperparameters
        model: Optional[nn.Module] = None
            Optional policy and value networks
        device: torch.device = torch.device('cpu')
            Torch device object where model is stored
    """
    def __init__(self, game: Game, kwargs: dict, model: Optional[nn.Module] = None, device: torch.device = torch.device("cpu")) -> None:
        self.game = game
        self.kwargs = kwargs
        self.model = model
        self.device = device

    @torch.no_grad()
    def search(self, state: np.ndarray) -> np.ndarray:
        """
        Apply MCTS starting from a specified game state.

        Args:
            state (np.ndarray): Array representation of current game state

        Returns:
            np.ndarray: Policy for current state after MCTS
        """
        # Create root node
        if self.model != None:
            root = Node(self.game, self.kwargs, state, is_alpha=True)
        else:
            root = Node(self.game, self.kwargs, state)

        # Run searches
        for _ in range(self.kwargs["n_searches"]):
            curr_node = root

            # select leaf node - find a node whose actions are not fully explored
            while curr_node.is_fully_expanded():
                # Move to child node
                curr_node = curr_node.select()

            # backpropagate if terminal node reached
            val, is_terminal = self.game.check_end_game(curr_node.state, curr_node.action_taken)
            val = self.game.get_opponent_value(val)

            if not is_terminal:
                policy = None
                if self.model != None: # get policy and value from model
                    enc_state = torch.tensor(self.game.encode_board(curr_node.state), dtype=torch.float32).unsqueeze(0).to(self.device)
                    with torch.inference_mode():
                        policy, val = self.model(enc_state)

                        # softmax on policy
                        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()

                    # set invalid moves to 0, rescale
                    valid_moves = self.game.get_valid_moves(curr_node.state)
                    policy *= valid_moves
                    policy /= policy.sum()

                    # get value
                    val = val.item()

                    # expand leaf node - take an action
                    curr_node.expand(policy)

                else:
                    # expand leaf node, take random action
                    curr_node = curr_node.expand()

                    # simulate by taking random actions
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