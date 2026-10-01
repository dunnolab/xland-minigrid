# jit-compatible variant of rgb_render.render, for rendering entire grids inside
# jitted or vmapped code. Tiles are pre-rendered once with the numpy renderer and
# then only indexed, so the output is identical to rgb_render.render pixel for pixel.
from __future__ import annotations

import os
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..benchmarks import load_bz2_pickle, save_bz2_pickle
from ..core.constants import NUM_COLORS, TILES_REGISTRY
from ..types import AgentState, GridState, IntOrArray
from .rgb_render import render_tile

CACHE_PATH = os.environ.get("XLAND_MINIGRID_CACHE", os.path.expanduser("~/.xland_minigrid"))
FORCE_RELOAD = os.environ.get("XLAND_MINIGRID_RELOAD_CACHE", False)

TILE_SIZE = 32
VIEW_SIZE = 7


def build_cache(tile_size: int = TILE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Pre-render every tile, with and without the field of view highlight.

    Returns two arrays indexed by [highlight, tile * NUM_COLORS + color]:
    plain tiles, and tiles with the agent drawn on top facing each direction.
    """
    tiles = np.asarray(TILES_REGISTRY)
    num_variants = tiles.shape[0] * tiles.shape[1]

    tile_cache = np.zeros((2, num_variants, tile_size, tile_size, 3), dtype=np.uint8)
    agent_cache = np.zeros((2, num_variants, 4, tile_size, tile_size, 3), dtype=np.uint8)

    for tile_id in range(tiles.shape[0]):
        for color_id in range(tiles.shape[1]):
            flat_idx = tile_id * tiles.shape[1] + color_id
            tile = tuple(tiles[tile_id, color_id])
            for highlight in (0, 1):
                tile_cache[highlight, flat_idx] = render_tile(
                    tile=tile,
                    agent_direction=None,
                    highlight=bool(highlight),
                    tile_size=tile_size,
                )
                for direction in range(4):
                    agent_cache[highlight, flat_idx, direction] = render_tile(
                        tile=tile,
                        agent_direction=direction,
                        highlight=bool(highlight),
                        tile_size=tile_size,
                    )

    return tile_cache, agent_cache


def load_cache(tile_size: int = TILE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Load the tile cache for a given tile size, building and saving it on first use."""
    cache_path = os.path.join(CACHE_PATH, f"grid_render_cache_{tile_size}")

    if not os.path.exists(cache_path) or FORCE_RELOAD:
        os.makedirs(CACHE_PATH, exist_ok=True)
        print("Building rendering cache, may take a while...")
        tile_cache, agent_cache = build_cache(tile_size=tile_size)
        print(f"Done. Cache is saved to {cache_path} and will be reused on consequent runs.")
        save_bz2_pickle({"tile_cache": tile_cache, "agent_cache": agent_cache}, cache_path)

    cache = load_bz2_pickle(cache_path)
    return cache["tile_cache"], cache["agent_cache"]


def highlight_mask(grid: GridState, agent: AgentState, view_size: IntOrArray = VIEW_SIZE) -> jax.Array:
    """Which grid cells fall inside the agent's field of view.

    Same region as rgb_render.get_highlight_mask, but computed with comparisons
    on grid coordinates instead of slicing into a padded array, so it traces.
    """
    direction = agent.direction.astype(jnp.int32)
    agent_y, agent_x = agent.position[0], agent.position[1]
    half = view_size // 2

    # top left corner of the view window, per direction
    top = jnp.stack(
        (
            agent_y - view_size + 1,
            agent_y - half,
            agent_y,
            agent_y - half,
        )
    )[direction]
    left = jnp.stack(
        (
            agent_x - half,
            agent_x,
            agent_x - half,
            agent_x - view_size + 1,
        )
    )[direction]

    ys, xs = jnp.meshgrid(jnp.arange(grid.shape[0]), jnp.arange(grid.shape[1]), indexing="ij")
    return (ys >= top) & (ys < top + view_size) & (xs >= left) & (xs < left + view_size)


def make_render(
    tile_size: int = TILE_SIZE,
    view_size: IntOrArray = VIEW_SIZE,
    cache: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> Callable[[GridState, Optional[AgentState]], jax.Array]:
    """Build a render function that can be jitted and vmapped.

    The returned function takes a grid and an optional agent and gives back a
    uint8 image of shape (grid_height * tile_size, grid_width * tile_size, 3),
    matching rgb_render.render for the same arguments. tile_size is fixed when
    the function is built, since it decides the shape of the cached tiles.
    """
    tiles, agent_tiles = load_cache(tile_size) if cache is None else cache
    tile_cache, agent_cache = jnp.asarray(tiles), jnp.asarray(agent_tiles)

    def render(grid: GridState, agent: Optional[AgentState] = None) -> jax.Array:
        height, width = grid.shape[0], grid.shape[1]
        flat_idxs = grid[:, :, 0].astype(jnp.int32) * NUM_COLORS + grid[:, :, 1].astype(jnp.int32)

        if agent is None:
            tiles = tile_cache[0][flat_idxs]
        else:
            mask = highlight_mask(grid, agent, view_size)
            tiles = jnp.where(mask[:, :, None, None, None], tile_cache[1][flat_idxs], tile_cache[0][flat_idxs])

            # the agent's own tile takes the highlight state of the cell it stands on
            agent_y, agent_x = agent.position[0], agent.position[1]
            agent_tile = agent_cache[
                mask[agent_y, agent_x].astype(jnp.int32),
                flat_idxs[agent_y, agent_x],
                agent.direction.astype(jnp.int32),
            ]
            tiles = tiles.at[agent_y, agent_x].set(agent_tile)

        # [height, width, tile_size, tile_size, 3] -> [height * tile_size, width * tile_size, 3]
        return tiles.transpose((0, 2, 1, 3, 4)).reshape(height * tile_size, width * tile_size, 3)

    return render
