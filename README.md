# Grid Worlds Gym
Different Gridworld implementations conforming to OpenAI gym interface. Used for developing Reinforcement Learning agents.

## Installation

`pip install git+https://github.com/drmaj/gridworldsgym#egg=gridworldsgym`

## Usage

```
$ import gym
$ import gridworldsgym
$ env = gym.make('GridWorld-v0')
```

## Type of Gridworlds

1. **GridWorld-v0**: Simple grid world from [Artificial Intelligence: A modern approach by Russel and Norvig](http://aima.cs.berkeley.edu/)

<div style="text-align:center;">

![](gifs/gridworld.gif)

</div>



2. **SlipperyGridWorld-v0**: Simple grid world with slippery actions from [Artificial Intelligence: A modern approach by Russel and Norvig](http://aima.cs.berkeley.edu/)
3. **WindyGridWorld-v0**: Windy Grid World from [Reinforcement Learning by Sutton and Barto](http://incompleteideas.net/book/RLbook2018.pdf)
4. **CliffGridWorld-v0**: Cliff Grid World from [Reinforcement Learning by Sutton and Barto](http://incompleteideas.net/book/RLbook2018.pdf)

## License
This project is licensed under the [GNU GENERAL PUBLIC LICENSE Version 3](https://www.gnu.org/licenses/gpl-3.0.en.html)

