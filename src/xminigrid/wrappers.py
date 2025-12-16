from __future__ import annotations

from typing import Any, Union

import jax

from .environment import Environment, EnvParamsT
from .types import EnvCarryT, IntOrArray, State, TimeStep


class Wrapper(Environment[EnvParamsT, EnvCarryT]):
    def __init__(self, env: Environment[EnvParamsT, EnvCarryT]):
        self._env = env

    # Question: what if wrapper adds new parameters to the dataclass?
    # Solution: do this after applying the wrapper:
    #   env_params = wrapped_env.default_params(**dataclasses.asdict(original_params))
    def default_params(self, **kwargs) -> EnvParamsT:
        return self._env.default_params(**kwargs)

    def num_actions(self, params: EnvParamsT) -> int:
        return self._env.num_actions(params)

    def observation_shape(self, params: EnvParamsT) -> Union[tuple[int, int, int], dict[str, Any]]:
        return self._env.observation_shape(params)

    def _generate_problem(self, params: EnvParamsT, key: jax.Array) -> State[EnvCarryT]:
        return self._env._generate_problem(params, key)

    def reset(self, params: EnvParamsT, key: jax.Array) -> TimeStep[EnvCarryT]:
        return self._env.reset(params, key)

    def step(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT], action: IntOrArray) -> TimeStep[EnvCarryT]:
        return self._env.step(params, timestep, action)

    def render(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT]):
        return self._env.render(params, timestep)


# gym and gymnasium style reset (on the same step with termination)
class GymAutoResetWrapper(Wrapper):
    def __auto_reset(self, params, timestep):
        key, _ = jax.random.split(timestep.state.key)
        reset_timestep = self._env.reset(params, key)

        timestep = timestep.replace(
            state=reset_timestep.state,
            observation=reset_timestep.observation,
        )
        return timestep

    # TODO: add last_obs somewhere in the timestep? add extras like in Jumanji?
    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        timestep = jax.lax.cond(
            timestep.last(),
            lambda: self.__auto_reset(params, timestep),
            lambda: timestep,
        )
        return timestep


# dm_env and envpool style reset (on the next step after termination)
class DmEnvAutoResetWrapper(Wrapper):
    def step(self, params, timestep, action):
        timestep = jax.lax.cond(
            timestep.last(),
            lambda: self._env.reset(params, timestep.state.key),
            lambda: self._env.step(params, timestep, action),
        )
        return timestep


# Yes, these are a bit stupid, but a tmp workaround to not write an actual system for spaces.
# May be, in the future, I will port the entire API to some existing one, like functional Gymnasium.
# For now, faster to do this stuff with dicts instead...
# NB: if you do not want to use this (due to the dicts as obs),
# just get needed parts from the original TimeStep and State dataclasses
class DirectionObservationWrapper(Wrapper):
    def observation_shape(self, params):
        base_shape = self._env.observation_shape(params)
        if isinstance(base_shape, dict):
            assert "img" in base_shape
            obs_shape = {**base_shape, **{"direction": 4}}
        else:
            obs_shape = {
                "img": self._env.observation_shape(params),
                "direction": 4,
            }
        return obs_shape

    def __extend_obs(self, timestep):
        direction = jax.nn.one_hot(timestep.state.agent.direction, num_classes=4)
        if isinstance(timestep.observation, dict):
            assert "img" in timestep.observation
            extended_obs = {
                **timestep.observation,
                **{"direction": direction},
            }
        else:
            extended_obs = {
                "img": timestep.observation,
                "direction": direction,
            }

        timestep = timestep.replace(observation=extended_obs)
        return timestep

    def reset(self, params, key):
        timestep = self._env.reset(params, key)
        timestep = self.__extend_obs(timestep)
        return timestep

    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        timestep = self.__extend_obs(timestep)
        return timestep


class RulesAndGoalsObservationWrapper(Wrapper):
    def observation_shape(self, params):
        base_shape = self._env.observation_shape(params)
        if isinstance(base_shape, dict):
            assert "img" in base_shape
            obs_shape = {
                **base_shape,
                **{
                    "goal_encoding": params.ruleset.goal.shape,
                    "rule_encoding": params.ruleset.rules.shape,
                },
            }
        else:
            obs_shape = {
                "img": self._env.observation_shape(params),
                "goal_encoding": params.ruleset.goal.shape,
                "rule_encoding": params.ruleset.rules.shape,
            }
        return obs_shape

    def __extend_obs(self, timestep):
        goal_encoding = timestep.state.goal_encoding
        rule_encoding = timestep.state.rule_encoding
        if isinstance(timestep.observation, dict):
            assert "img" in timestep.observation
            extended_obs = {
                **timestep.observation,
                **{
                    "goal_encoding": goal_encoding,
                    "rule_encoding": rule_encoding,
                },
            }
        else:
            extended_obs = {
                "img": timestep.observation,
                "goal_encoding": goal_encoding,
                "rule_encoding": rule_encoding,
            }

        timestep = timestep.replace(observation=extended_obs)
        return timestep

    def reset(self, params, key):
        timestep = self._env.reset(params, key)
        timestep = self.__extend_obs(timestep)
        return timestep

    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        timestep = self.__extend_obs(timestep)
        return timestep
