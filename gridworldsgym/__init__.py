from gym.envs.registration import register

register(id='GridWorld-v0',
         entry_point='gridworldsgym.envs:GridWorldV0')

register(id='SlipperyGridWorld-v0',
         entry_point='gridworldsgym.envs:GridWorldV0',
         kwargs={'slippery': True}
         )

register(id='WindyGridWorld-v0',
         entry_point='gridworldsgym.envs:WindyGridWorldV0')

register(id='CliffGridWorld-v0',
         entry_point='gridworldsgym.envs:CliffGridWorldV0')
