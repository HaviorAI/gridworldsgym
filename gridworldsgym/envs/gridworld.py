import numpy as np

from gridworldsgym.envs.discrete import FiniteStateMDP

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridWorldV0(FiniteStateMDP):
    """ Grid world example from chapter 17 of Artificial Intelligence: A modern approach ()
        Terminal states are at (1, 3) and (2, 3) with a reward of -1 and +1 respectively. If
        slippery = True, then there's an 80% chance that the desired action will result the
        desired move, and a 10% it will result in a slip to the left or to the right of the
        desired direction. If slippery = False, the desired action will be taken without any
        slipping.
    """

    def render(self, mode='human'):
        pass

    def __init__(self, width=4, height=3, slippery=False):
        self.width = width
        self.height = height
        self.slippery = slippery
        num_states = width * height
        num_actions = 4
        isd = np.zeros(num_states)
        # this starts the agent in state 0
        isd[0] = 1
        self.terminal_states = [(1, 3), (2, 3)]
        self.illegal_states = [(1, 1)]
        super(GridWorldV0, self).__init__(num_states, num_actions, isd=isd)
        self.transitions = self._generate_transitions()
        self.rewards = self._generate_rewards()

    def _to_state(self, row, col):
        return row * self.width + col

    def to_row_col(self, state):
        row = state // self.width
        col = state - row * self.width
        return row, col

    def _move(self, row, col, action):
        if action == LEFT:
            col = max(col - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.height - 1)
        elif action == RIGHT:
            col = min(col + 1, self.width - 1)
        elif action == UP:
            row = max(row - 1, 0)
        return row, col

    def _generate_transitions(self):
        transitions = {s: {a: [] for a in range(self.num_actions)} for s in range(self.num_states)}
        for row in range(self.height):
            for col in range(self.width):
                state = self._to_state(row, col)
                for action in range(self.num_actions):
                    if (row, col) in self.terminal_states:
                        transitions[state][action].append((0.0, state, True))
                    else:
                        if self.slippery:
                            action_probs = [0.1, 0.8, 0.1]
                            actions = [(action - 1) % self.num_actions, action, (action + 1) % self.num_actions]
                        else:
                            action_probs = [1.0]
                            actions = [action]
                        for i in range(len(actions)):
                            new_row, new_col = self._move(row, col, actions[i])
                            new_state = self._to_state(new_row, new_col)
                            if (new_row, new_col) in self.illegal_states:
                                new_state = state
                            done = (new_row, new_col) in self.terminal_states
                            transitions[state][action].append((action_probs[i], new_state, done))
        return transitions

    def _generate_rewards(self):
        rewards = -0.04 * np.ones(self.num_states)
        for illegal_state in self.illegal_states:
            rewards[self._to_state(*illegal_state)] = None
        term_state_1 = self._to_state(*self.terminal_states[0])
        term_state_2 = self._to_state(*self.terminal_states[1])
        rewards[term_state_1] = -1.0
        rewards[term_state_2] = 1.0
        return rewards

    def _check_done(self):
        return self.to_row_col(self.state) in self.terminal_states

    def P(self, state, action):
        return self.transitions[state][action]

    def R(self, state, action=None):
        return self.rewards[state]
