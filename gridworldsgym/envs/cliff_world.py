import numpy as np
from gridworldsgym.envs import GridWorldV0


class CliffGridWorldV0(GridWorldV0):
    # TODO: Add doc string
    def __init__(self, width=12, height=4, slippery=False):
        super(CliffGridWorldV0, self).__init__(width, height, slippery=slippery)
        self.isd = np.zeros(self.num_states)
        self.isd[self._to_state(3, 0)] = 1.0
        self.terminal_states = [(3, i) for i in range(1, self.width)]
        self.illegal_states = []
        self.transitions = self._generate_transitions()
        self.rewards = self._generate_rewards()

    def _generate_rewards(self):
        rewards = -1.0 * np.ones(self.num_states)
        for i in range(1, self.width - 1):
            rewards[self._to_state(3, i)] = -100
        return rewards
