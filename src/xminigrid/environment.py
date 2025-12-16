from __future__ import annotations

import abc
from typing import Any, Generic, Optional, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from .core.actions import take_action
from .core.constants import NUM_ACTIONS, NUM_LAYERS
from .core.goals import check_goal
from .core.observation import transparent_field_of_view
from .core.rules import check_rule
from .rendering.rgb_render import render as rgb_render
from .rendering.text_render import render as text_render
from .types import EnvCarryT, IntOrArray, State, StepType, TimeStep


class EnvParams(struct.PyTreeNode):
    # WARN: pytree_node=False, so you CAN NOT vmap on them!
    # You can add pytree node params, but be careful and
    # test that your code will work under jit.
    # Spoiler: probably it will not :(
    height: int = struct.field(pytree_node=False, default=9)
    width: int = struct.field(pytree_node=False, default=9)
    view_size: int = struct.field(pytree_node=False, default=7)
    max_steps: Optional[int] = struct.field(pytree_node=False, default=None)
    render_mode: str = struct.field(pytree_node=False, default="rgb_array")


EnvParamsT = TypeVar("EnvParamsT", bound="EnvParams")


class Environment(abc.ABC, Generic[EnvParamsT, EnvCarryT]):
    @abc.abstractmethod
    def default_params(self, **kwargs: Any) -> EnvParamsT: ...

    def num_actions(self, params: EnvParamsT) -> int:
        return int(NUM_ACTIONS)

    def observation_shape(self, params: EnvParamsT) -> tuple[int, int, int] | dict[str, Any]:
        return params.view_size, params.view_size, NUM_LAYERS

    @abc.abstractmethod
    def _generate_problem(self, params: EnvParamsT, key: jax.Array) -> State[EnvCarryT]: ...

    def reset(self, params: EnvParamsT, key: jax.Array) -> TimeStep[EnvCarryT]:
        state = self._generate_problem(params, key)
        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=transparent_field_of_view(state.grid, state.agent, params.view_size, params.view_size),
        )
        return timestep

    # Why timestep + state at once, and not like in Jumanji? To be able to do autoresets in gym and envpools styles
    def step(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT], action: IntOrArray) -> TimeStep[EnvCarryT]:
        new_grid, new_agent, changed_position = take_action(timestep.state.grid, timestep.state.agent, action)
        new_grid, new_agent = check_rule(timestep.state.rule_encoding, new_grid, new_agent, action, changed_position)

        new_state = timestep.state.replace(
            grid=new_grid,
            agent=new_agent,
            step_num=timestep.state.step_num + 1,
        )
        new_observation = transparent_field_of_view(new_state.grid, new_state.agent, params.view_size, params.view_size)

        # checking for termination or truncation, choosing step type
        terminated = check_goal(new_state.goal_encoding, new_state.grid, new_state.agent, action, changed_position)

        assert params.max_steps is not None
        truncated = jnp.equal(new_state.step_num, params.max_steps)

        reward = jax.lax.select(terminated, 1.0 - 0.9 * (new_state.step_num / params.max_steps), 0.0)

        step_type = jax.lax.select(terminated | truncated, StepType.LAST, StepType.MID)
        discount = jax.lax.select(terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        timestep = TimeStep(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_observation,
        )
        return timestep

    def render(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT]) -> np.ndarray | str:
        if params.render_mode == "rgb_array":
            return rgb_render(np.asarray(timestep.state.grid), timestep.state.agent, params.view_size)
        elif params.render_mode == "rich_text":
            return text_render(timestep.state.grid, timestep.state.agent)
        else:
            raise RuntimeError("Unknown render mode. Should be one of: ['rgb_array', 'rich_text']")
