import gymnasium as gym
import jax
import numpy as np
from gymnasium import spaces

from xminigrid.environment import Environment, EnvParams
from xminigrid.types import TimeStep


class JaxWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array", "rich_text"]}

    def __init__(self, env: Environment, params: EnvParams):
        super().__init__()
        self._key = jax.random.key(0)
        self._jit_reset = jax.jit(env.reset)
        self._jit_step = jax.jit(env.step)
        self._render = env.render  # not jittable
        self._timestep: TimeStep | None = None
        self._params = params

        # spaces
        self.action_space = spaces.Discrete(env.num_actions(params))
        observation_space = env.observation_shape(params)
        if isinstance(observation_space, dict):
            observation_space_dict: dict[str, spaces.Space] = {
                key: spaces.Discrete(shape)
                if isinstance(shape, int)
                else spaces.Box(low=np.inf, high=255, shape=shape, dtype=np.uint8)
                for key, shape in observation_space.items()
            }
            self.observation_space = spaces.Dict(observation_space_dict)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=observation_space, dtype=np.uint8)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # set _np_random
        if options is not None:
            self._params = type(self._params)(**options)
        rng = jax.numpy.frombuffer(self.np_random.bytes(4), jax.numpy.int32)
        self._timestep = self._jit_reset(self._params, jax.random.key(rng[0]))
        return jax.tree.map(np.asarray, self._timestep.observation), {}

    def step(self, action):
        self._timestep: TimeStep = self._jit_step(self._params, self._timestep, action)
        term = jax.numpy.isclose(self._timestep.discount, 0).item()
        return (
            jax.tree.map(np.asarray, self._timestep.observation),
            self._timestep.reward.item(),
            term,
            self._timestep.last().item() and not term,
            {},
        )

    def render(self):
        return self._render(self._params, self._timestep)
