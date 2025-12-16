from __future__ import annotations

from typing import Generic, TypeVar, Union

import jax
import jax.numpy as jnp
from flax import struct
from typing_extensions import TypeAlias

from .core.constants import TILES_REGISTRY, Colors, Tiles


class RuleSet(struct.PyTreeNode):
    goal: jax.Array
    rules: jax.Array
    init_tiles: jax.Array


GridState: TypeAlias = jax.Array
Tile: TypeAlias = jax.Array
IntOrArray: TypeAlias = Union[int, jax.Array]
EnvCarryT = TypeVar("EnvCarryT")


class AgentState(struct.PyTreeNode):
    position: jax.Array = struct.field(default_factory=lambda: jnp.asarray((0, 0)))
    direction: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0))
    pocket: jax.Array = struct.field(default_factory=lambda: TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY])


class EnvCarry(struct.PyTreeNode): ...


class State(struct.PyTreeNode, Generic[EnvCarryT]):
    key: jax.Array
    step_num: jax.Array

    grid: GridState
    agent: AgentState
    goal_encoding: jax.Array
    rule_encoding: jax.Array

    carry: EnvCarryT


class StepType(jnp.uint8):
    FIRST: jax.Array = jnp.asarray(0, dtype=jnp.uint8)
    MID: jax.Array = jnp.asarray(1, dtype=jnp.uint8)
    LAST: jax.Array = jnp.asarray(2, dtype=jnp.uint8)


class TimeStep(struct.PyTreeNode, Generic[EnvCarryT]):
    state: State[EnvCarryT]

    step_type: StepType
    reward: jax.Array
    discount: jax.Array
    observation: jax.Array | dict[str, jax.Array]

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST
