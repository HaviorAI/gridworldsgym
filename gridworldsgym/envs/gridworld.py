import numpy as np

from gridworldsgym.envs.discrete import FiniteStateMDP

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

NORTH = 0
SOUTH = np.math.pi
EAST = -np.math.pi / 2
WEST = -3 * np.math.pi / 2


class GridWorldV0(FiniteStateMDP):
    """ Grid world example from chapter 17 of Artificial Intelligence: A modern approach ()
        Terminal states are at (1, 3) and (2, 3) with a reward of -1 and +1 respectively. If
        slippery = True, then there's an 80% chance that the desired action will result the
        desired move, and a 10% it will result in a slip to the left or to the right of the
        desired direction. If slippery = False, the desired action will be taken without any
        slipping.
    """

    def render(self, mode='human'):
        # TODO: add option to show cumulative rewards
        # TODO: add option to show value function
        screen_width = 100 * self.width
        screen_height = 100 * self.height

        self._set_heading(self.last_action)

        def get_x_y(row, col):
            return col * 100 + 50, screen_height - (row * 100 + 50)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            for i in range(1, self.width):
                line = rendering.Line((i * 100, 0), (i * 100, screen_height))
                line.set_color(0, 0, 0)
                self.viewer.add_geom(line)
            for j in range(1, self.height):
                line = rendering.Line((0, j * 100), (screen_width, j * 100))
                line.set_color(0, 0, 0)
                self.viewer.add_geom(line)

            for row, col in self.goal_states:
                goal = rendering.FilledPolygon([(-49, -49), (-49, 49), (49, 49), (49, -49)])
                goal.set_color(0.0, 1.0, 0.0)
                goal_transform = rendering.Transform()
                goal.add_attr(goal_transform)
                new_x, new_y = get_x_y(row, col)
                goal_transform.set_translation(new_x, new_y)
                self.viewer.add_geom(goal)

            for row, col in self.terminal_states:
                goal = rendering.FilledPolygon([(-49, -49), (-49, 49), (49, 49), (49, -49)])
                goal.set_color(1.0, 0.0, 0.0)
                goal_transform = rendering.Transform()
                goal.add_attr(goal_transform)
                new_x, new_y = get_x_y(row, col)
                goal_transform.set_translation(new_x, new_y)
                self.viewer.add_geom(goal)

            agent_size = 50
            l, r, t, b = -agent_size / 2, agent_size / 2, agent_size / 2, -agent_size / 2
            x, y = get_x_y(0, 0)
            self.agent_transform = rendering.Transform(translation=(x, y))
            agent = rendering.FilledPolygon([(l, b), (0.0, t), (0.0 * r, t), (r, b)])
            agent.set_color(95 / 256, 125 / 256, 153 / 256)
            agent.add_attr(self.agent_transform)
            self.viewer.add_geom(agent)

        row, col = self.to_row_col(self.state)
        new_x, new_y = get_x_y(row, col)
        self.agent_transform.set_translation(new_x, new_y)
        self.agent_transform.set_rotation(self.heading)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def __init__(self, width=4, height=3, slippery=False):
        self.width = width
        self.height = height
        self.slippery = slippery
        num_states = width * height
        num_actions = 4
        isd = np.zeros(num_states)
        # this starts the agent in state 0
        isd[0] = 1
        self.goal_states = [(2, 3)]
        self.terminal_states = [(1, 3)]
        self.illegal_states = [(1, 1)]
        super(GridWorldV0, self).__init__(num_states, num_actions, isd=isd)
        self.transitions = self._generate_transitions()
        self.rewards = self._generate_rewards()
        self.viewer = None
        self.heading = NORTH

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

    def _set_heading(self, action):
        if action == LEFT:
            self.heading = WEST
        elif action == DOWN:
            self.heading = SOUTH
        elif action == RIGHT:
            self.heading = EAST
        elif action == UP:
            self.heading = NORTH

    def _generate_transitions(self):
        transitions = {s: {a: [] for a in range(self.num_actions)} for s in range(self.num_states)}
        for row in range(self.height):
            for col in range(self.width):
                state = self._to_state(row, col)
                for action in range(self.num_actions):
                    if (row, col) in self.terminal_states or (row, col) in self.goal_states:
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
                            done = (new_row, new_col) in self.terminal_states or (new_row, new_col) in self.goal_states
                            transitions[state][action].append((action_probs[i], new_state, done))
        return transitions

    def _generate_rewards(self):
        rewards = -0.04 * np.ones(self.num_states)
        for illegal_state in self.illegal_states:
            rewards[self._to_state(*illegal_state)] = None
        term_state = self._to_state(*self.terminal_states[0])
        goal_state = self._to_state(*self.goal_states[0])
        rewards[term_state] = -1.0
        rewards[goal_state] = 1.0
        return rewards

    def _check_done(self):
        return self.to_row_col(self.state) in self.terminal_states

    def P(self, state, action):
        return self.transitions[state][action]

    def R(self, state, action=None):
        return self.rewards[state]
