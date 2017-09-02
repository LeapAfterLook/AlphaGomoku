import numpy as np
import time
from gomoku_manager import GomokuManager


class MCNode:

    def __init__(self, value=0, prev_move=None, state=None, parent=None):
        self.value = value
        self.prev_move = prev_move # one-hot encoded
        self.state = state
        self.num_visit = 0
        self.children = []
        self.parent = parent

    # __lt__ = lambda self, other: self.value < other.value
    #def __lt__(self, other):
    #    return self.value > other.value #역순

    # next turn
    def calculate_turn(self):
        # TODO: implement this
        return 'black'

    def add_child(self, child):
        self.children.append(child)

    def forward(self, policy):
        p = np.array(policy.get_next_move_distribution(self.state))
        q = np.array([0 for i in range(225)])

        for child in self.children:
            q += np.array(child.prev_move) * child.value
            action_index = np.argmax(child.prev_move)
            p[action_index] /= ( 1 + child.num_visit )

        next_move_index = np.argmax(p+q)
        next_move = [0 for i in range(225)]
        next_move[next_move_index] = 1

        for child in self.children:
            if next_move == child.prev_move
                return child.forward(policy)

        return self, next_move

    def update_and_backprop(self, value):
        num_visit = self.num_visit
        self.value = (num_visit/(num_visit+1)) * self.value + (1/(num_visit+1)) * value
        self.num_visit += 1

        if not self.is_root():
            self.parent.update_and_backprop(value=value)

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None


class MCTree:

    def __init__(self, game_manager=None, init_state=np.zeros(15,15), my_policy=None, enemy_policy=None, time_limit=20):
        self.root = MCNode(init_state)
        self.game_manager = GomokuManager()
        self.my_policy = my_policy
        self.enemy_policy = enemy_policy
        self.time_limit = time_limit
        self.expand()

    def expand_one_time(self):
        best_desc, next_move = self.root.forward(self.my_policy)

        self.game_manager.set(curr_state=best_desc.state)
        next_state, result = self.game_manager.next_state_with_montecarlo_result(next_move=next_move,
                                                                                 my_policy=self.my_policy,
                                                                                 enemy_policy=self.enemy_policy)

        if self.root.turn() == 'black':
            value = (-1) * result
        else:
            value = result

        new_leaf = MCNode(parent=best_desc, state=next_state, prev_move=next_move)
        best_desc.add_child(new_leaf)
        new_leaf.update_and_backprop(value=value)

    def expand(self):
        start_time = time.time()

        while time.time() - start_time < self.time_limit:
            self.expand_one_time()

    def best_choice(self):
        return self.root.children.get()