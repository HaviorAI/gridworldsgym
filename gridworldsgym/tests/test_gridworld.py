import gym
import pytest
import gridworldsgym


@pytest.fixture
def gridworld_env():
    env = gym.make('GridWorld-v0')
    return env


def test_gridworld_creation(gridworld_env):
    env = gridworld_env
    # TODO: fix this test
    assert True


def test_gridworld_step(gridworld_env):
    env = gridworld_env
    env.step(env.action_space.sample())
    # TODO: fix this test
    assert True