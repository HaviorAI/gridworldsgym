import numpy as np

from gridworldsgym.envs import GridWorldV0
from gridworldsgym.envs.gridworld import UP, RIGHT, DOWN, LEFT


class WindyGridWorldV0(GridWorldV0):
    # TODO: Add doc string
    def __init__(self, width=10, height=7):
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        super(WindyGridWorldV0, self).__init__(width, height)
        self.isd = np.zeros(self.num_states)
        self.isd[30] = 1.0
        self.goal_states = [(3, 7)]
        self.terminal_states = []
        self.illegal_states = []
        self.transitions = self._generate_transitions()
        self.rewards = self._generate_rewards()

    def _move(self, row, col, action):
        wind = self.wind[col]
        if action == LEFT:
            col = max(col - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.height - 1)
        elif action == RIGHT:
            col = min(col + 1, self.width - 1)
        elif action == UP:
            row = max(row - 1, 0)
        row = max(row - wind, 0)
        return row, col

    def _generate_transitions(self):
        transitions = {s: {a: [] for a in range(self.num_actions)} for s in range(self.num_states)}
        for row in range(self.height):
            for col in range(self.width):
                state = self._to_state(row, col)
                for action in range(self.num_actions):
                    if (row, col) in self.terminal_states or (row, col) in self.goal_states:
                        transitions[state][action].append((0.0, state, True))
                    else:
                        action_probs = [1.0]
                        actions = [action]
                        for i in range(len(actions)):
                            new_row, new_col = self._move(row, col, actions[i])
                            new_state = self._to_state(new_row, new_col)
                            done = (new_row, new_col) in self.terminal_states or (new_row, new_col) in self.goal_states
                            transitions[state][action].append((action_probs[i], new_state, done))
        return transitions

    def _generate_rewards(self):
        rewards = -1.0 * np.ones(self.num_states)
        return rewards
