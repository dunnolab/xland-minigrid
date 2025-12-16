from __future__ import annotations

import abc
from typing import Union

import jax
import jax.numpy as jnp
from flax import struct

from ..types import AgentState, GridState
from .grid import equal, get_neighbouring_tiles, pad_along_axis

MAX_GOAL_ENCODING_LEN = 4 + 1  # for idx
NUM_GOALS = 15


def check_goal(
    encoding: jax.Array, grid: GridState, agent: AgentState, action: Union[int, jax.Array], position: jax.Array
) -> jax.Array:
    check = jax.lax.switch(
        encoding[0],
        (
            # empty goal first, we use them as paddings during benchmark generation
            lambda: EmptyGoal.decode(encoding)(grid, agent, action, position),
            lambda: AgentHoldGoal.decode(encoding)(grid, agent, action, position),
            lambda: AgentOnTileGoal.decode(encoding)(grid, agent, action, position),
            lambda: AgentNearGoal.decode(encoding)(grid, agent, action, position),
            lambda: TileNearGoal.decode(encoding)(grid, agent, action, position),
            lambda: TileOnPositionGoal.decode(encoding)(grid, agent, action, position),
            lambda: AgentOnPositionGoal.decode(encoding)(grid, agent, action, position),
            # goals for the extended benchmarks
            lambda: TileNearUpGoal.decode(encoding)(grid, agent, action, position),
            lambda: TileNearRightGoal.decode(encoding)(grid, agent, action, position),
            lambda: TileNearDownGoal.decode(encoding)(grid, agent, action, position),
            lambda: TileNearLeftGoal.decode(encoding)(grid, agent, action, position),
            lambda: AgentNearUpGoal.decode(encoding)(grid, agent, action, position),
            lambda: AgentNearRightGoal.decode(encoding)(grid, agent, action, position),
            lambda: AgentNearDownGoal.decode(encoding)(grid, agent, action, position),
            lambda: AgentNearLeftGoal.decode(encoding)(grid, agent, action, position),
        ),
    )
    return check


# should I inherit from abc.ABC?
class BaseGoal(struct.PyTreeNode):
    @abc.abstractmethod
    def __call__(
        self, grid: GridState, agent: AgentState, action: Union[int, jax.Array], position: jax.Array
    ) -> jax.Array: ...

    @classmethod
    @abc.abstractmethod
    def decode(cls, encoding: jax.Array) -> BaseGoal: ...

    @abc.abstractmethod
    def encode(self) -> jax.Array: ...


class EmptyGoal(BaseGoal):
    def __call__(self, grid, agent, action, position):
        return jnp.asarray(False)

    @classmethod
    def decode(cls, encoding):
        return cls()

    def encode(self):
        return jnp.zeros(MAX_GOAL_ENCODING_LEN, dtype=jnp.uint8)


class AgentHoldGoal(BaseGoal):
    tile: jax.Array

    def __call__(self, grid, agent, action, position):
        check = jax.lax.select(jnp.equal(action, 3), equal(agent.pocket, self.tile), jnp.asarray(False))
        return check

    @classmethod
    def decode(cls, encoding):
        return cls(tile=encoding[1:3])

    def encode(self):
        encoding = jnp.hstack([jnp.asarray(1), self.tile], dtype=jnp.uint8)
        return pad_along_axis(encoding, MAX_GOAL_ENCODING_LEN)


class AgentOnTileGoal(BaseGoal):
    tile: jax.Array

    def __call__(self, grid, agent, action, position):
        check = jax.lax.select(
            jnp.equal(action, 0), equal(grid[agent.position[0], agent.position[1]], self.tile), jnp.asarray(False)
        )
        return check

    @classmethod
    def decode(cls, encoding):
        return cls(tile=encoding[1:3])

    def encode(self):
        encoding = jnp.hstack([jnp.asarray(2), self.tile], dtype=jnp.uint8)
        return pad_along_axis(encoding, MAX_GOAL_ENCODING_LEN)


class AgentNearGoal(BaseGoal):
    tile: jax.Array

    def __call__(self, grid, agent, action, position):
        def _check_fn():
            up, right, down, left = get_neighbouring_tiles(grid, agent.position[0], agent.position[1])
            check = equal(up, self.tile) | equal(right, self.tile) | equal(down, self.tile) | equal(left, self.tile)

            return check

        check = jax.lax.select(jnp.equal(action, 0), _check_fn(), jnp.asarray(False))
        return check

    @classmethod
    def decode(cls, encoding):
        return cls(tile=encoding[1:3])

    def encode(self):
        encoding = jnp.hstack([jnp.asarray(3), self.tile], dtype=jnp.uint8)
        return pad_along_axis(encoding, MAX_GOAL_ENCODING_LEN)


class TileNearGoal(BaseGoal):
    tile_a: jax.Array
    tile_b: jax.Array

    def __call__(self, grid, agent, action, position):
        tile_a = self.tile_a
        tile_b = self.tile_b

        def _check_fn():
            tile = grid[position[0], position[1]]
            up, right, down, left = get_neighbouring_tiles(grid, position[0], position[1])

            check = jax.lax.select(
                jnp.logical_or(equal(tile, tile_a), equal(tile, tile_b)),
                jax.lax.select(
                    equal(tile, tile_a),
                    equal(up, tile_b) | equal(right, tile_b) | equal(down, tile_b) | equal(left, tile_b),
                    equal(up, tile_a) | equal(right, tile_a) | equal(down, tile_a) | equal(left, tile_a),
                ),
                jnp.asarray(False),
            )
            return check

        check = jax.lax.select(jnp.equal(action, 4), _check_fn(), jnp.asarray(False))
        return check

    @classmethod
    def decode(cls, encoding):
        return cls(tile_a=encoding[1:3], tile_b=encoding[3:5])

    def encode(self):
        encoding = jnp.hstack([jnp.asarray(4), self.tile_a, self.tile_b], dtype=jnp.uint8)
        return pad_along_axis(encoding, MAX_GOAL_ENCODING_LEN)


class TileOnPositionGoal(BaseGoal):
    tile: jax.Array
    position: jax.Array

    def __call__(self, grid, agent, action, position):
        check = jnp.array_equal(grid[self.position[0], self.position[1]], self.tile)
        return check

    @classmethod
    def decode(cls, encoding):
        return cls(tile=encoding[1:3], position=encoding[3:5])

    def encode(self):
        # WARN: uint8 type means that max grid size currently is 255
        encoding = jnp.hstack([jnp.asarray(5), self.tile, self.position], dtype=jnp.uint8)
        return pad_along_axis(encoding, MAX_GOAL_ENCODING_LEN)


class AgentOnPositionGoal(BaseGoal):
    position: jax.Array

    def __call__(self, grid, agent, action, position):
        check = jnp.array_equal(agent.position, self.position)
        return check

    @classmethod
    def decode(cls, encoding):
        return cls(position=encoding[1:3])

    def encode(self):
        encoding = jnp.hstack([jnp.asarray(6), self.position], dtype=jnp.uint8)
        return pad_along_axis(encoding, MAX_GOAL_ENCODING_LEN)


class TileNearUpGoal(BaseGoal):
    tile_a: jax.Array
    tile_b: jax.Array

    def __call__(self, grid, agent, action, position):
        y, x = position
        tile = grid[y, x]

        def _check_fn():
            up, _, down, _ = get_neighbouring_tiles(grid, y, x)
            check = jnp.logical_or(
                equal(tile, self.tile_b) & equal(down, self.tile_a), equal(tile, self.tile_a) & equal(up, self.tile_b)
            )
            return check

        check = jax.lax.select(
            jnp.equal(action, 4) & (equal(tile, self.tile_a) | equal(tile, self.tile_b)),
            _check_fn(),
            jnp.asarray(False),
        )
        return check

    @classmethod
    def decode(cls, encoding):
        return cls(tile_a=encoding[1:3], tile_b=encoding[3:5])

    def encode(self):
        encoding = jnp.hstack([jnp.asarray(7), self.tile_a, self.tile_b])
        return pad_along_axis(encoding, MAX_GOAL_ENCODING_LEN)


class TileNearRightGoal(BaseGoal):
    tile_a: jax.Array
    tile_b: jax.Array

    def __call__(self, grid, agent, action, position):
        y, x = position
        tile = grid[y, x]

        def _check_fn():
            _, right, _, left = get_neighbouring_tiles(grid, y, x)
            check = jnp.logical_or(
                equal(tile, self.tile_b) & equal(left, self.tile_a),
                equal(tile, self.tile_a) & equal(right, self.tile_b),
            )
            return check

        check = jax.lax.select(
            jnp.equal(action, 4) & (equal(tile, self.tile_a) | equal(tile, self.tile_b)),
            _check_fn(),
            jnp.asarray(False),
        )
        return check

    @classmethod
    def decode(cls, encoding):
        return cls(tile_a=encoding[1:3], tile_b=encoding[3:5])

    def encode(self):
        encoding = jnp.hstack([jnp.asarray(8), self.tile_a, self.tile_b])
        return pad_along_axis(encoding, MAX_GOAL_ENCODING_LEN)


class TileNearDownGoal(BaseGoal):
    tile_a: jax.Array
    tile_b: jax.Array

    def __call__(self, grid, agent, action, position):
        y, x = position
        tile = grid[y, x]

        def _check_fn():
            up, _, down, _ = get_neighbouring_tiles(grid, y, x)
            check = jnp.logical_or(
                equal(tile, self.tile_b) & equal(up, self.tile_a), equal(tile, self.tile_a) & equal(down, self.tile_b)
            )
            return check

        check = jax.lax.select(
            jnp.equal(action, 4) & (equal(tile, self.tile_a) | equal(tile, self.tile_b)),
            _check_fn(),
            jnp.asarray(False),
        )
        return check

    @classmethod
    def decode(cls, encoding):
        return cls(tile_a=encoding[1:3], tile_b=encoding[3:5])

    def encode(self):
        encoding = jnp.hstack([jnp.asarray(9), self.tile_a, self.tile_b])
        return pad_along_axis(encoding, MAX_GOAL_ENCODING_LEN)


class TileNearLeftGoal(BaseGoal):
    tile_a: jax.Array
    tile_b: jax.Array

    def __call__(self, grid, agent, action, position):
        y, x = position
        tile = grid[y, x]

        def _check_fn():
            _, right, _, left = get_neighbouring_tiles(grid, y, x)
            check = jnp.logical_or(
                equal(tile, self.tile_b) & equal(right, self.tile_a),
                equal(tile, self.tile_a) & equal(left, self.tile_b),
            )
            return check

        check = jax.lax.select(
            jnp.equal(action, 4) & (equal(tile, self.tile_a) | equal(tile, self.tile_b)),
            _check_fn(),
            jnp.asarray(False),
        )
        return check

    @classmethod
    def decode(cls, encoding):
        return cls(tile_a=encoding[1:3], tile_b=encoding[3:5])

    def encode(self):
        encoding = jnp.hstack([jnp.asarray(10), self.tile_a, self.tile_b])
        return pad_along_axis(encoding, MAX_GOAL_ENCODING_LEN)


class AgentNearUpGoal(BaseGoal):
    tile: jax.Array

    def __call__(self, grid, agent, action, position):
        def _check_fn():
            up, _, _, _ = get_neighbouring_tiles(grid, agent.position[0], agent.position[1])
            check = equal(up, self.tile)
            return check

        check = jax.lax.select(
            jnp.equal(action, 0) | jnp.equal(action, 4),
            _check_fn(),
            jnp.asarray(False),
        )
        return check

    @classmethod
    def decode(cls, encoding):
        return cls(tile=encoding[1:3])

    def encode(self):
        encoding = jnp.hstack([jnp.asarray(11), self.tile])
        return pad_along_axis(encoding, MAX_GOAL_ENCODING_LEN)


class AgentNearRightGoal(BaseGoal):
    tile: jax.Array

    def __call__(self, grid, agent, action, position):
        def _check_fn():
            _, right, _, _ = get_neighbouring_tiles(grid, agent.position[0], agent.position[1])
            check = equal(right, self.tile)
            return check

        check = jax.lax.select(
            jnp.equal(action, 0) | jnp.equal(action, 4),
            _check_fn(),
            jnp.asarray(False),
        )
        return check

    @classmethod
    def decode(cls, encoding):
        return cls(tile=encoding[1:3])

    def encode(self):
        encoding = jnp.hstack([jnp.asarray(12), self.tile])
        return pad_along_axis(encoding, MAX_GOAL_ENCODING_LEN)


class AgentNearDownGoal(BaseGoal):
    tile: jax.Array

    def __call__(self, grid, agent, action, position):
        def _check_fn():
            _, _, down, _ = get_neighbouring_tiles(grid, agent.position[0], agent.position[1])
            check = equal(down, self.tile)
            return check

        check = jax.lax.select(
            jnp.equal(action, 0) | jnp.equal(action, 4),
            _check_fn(),
            jnp.asarray(False),
        )
        return check

    @classmethod
    def decode(cls, encoding):
        return cls(tile=encoding[1:3])

    def encode(self):
        encoding = jnp.hstack([jnp.asarray(13), self.tile])
        return pad_along_axis(encoding, MAX_GOAL_ENCODING_LEN)


class AgentNearLeftGoal(BaseGoal):
    tile: jax.Array

    def __call__(self, grid, agent, action, position):
        def _check_fn():
            _, _, _, left = get_neighbouring_tiles(grid, agent.position[0], agent.position[1])
            check = equal(left, self.tile)
            return check

        check = jax.lax.select(
            jnp.equal(action, 0) | jnp.equal(action, 4),
            _check_fn(),
            jnp.asarray(False),
        )
        return check

    @classmethod
    def decode(cls, encoding):
        return cls(tile=encoding[1:3])

    def encode(self):
        encoding = jnp.hstack([jnp.asarray(14), self.tile])
        return pad_along_axis(encoding, MAX_GOAL_ENCODING_LEN)
