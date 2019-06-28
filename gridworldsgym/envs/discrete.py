import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class FiniteStateMDP(gym.Env):
    def __init__(self, num_states, num_actions, isd=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.np_random = None
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)
        self.isd = isd
        self.last_action = None
        self.seed()
        self.reward = []
        self.state = self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        if self.isd is not None:
            self.state = self._sample(self.isd)
            self.last_action = None
            return self.state

    def step(self, action):
        transitions = self.P(self.state, action)
        index = self._sample([t[0] for t in transitions])
        prob, state, done = transitions[index]
        reward = self.R(state, action)
        self.state = state
        self.last_action = action
        return state, reward, done, {"prob": prob}

    def render(self, mode='human'):
        raise NotImplementedError

    def _sample(self, probs):
        probs = np.asarray(probs)
        csprobs = np.cumsum(probs)
        return (csprobs > self.np_random.rand()).argmax()

    def P(self, state, action):
        raise NotImplementedError

    def R(self, state, action=None):
        raise NotImplementedError

    def _check_done(self):
        raise NotImplementedError
