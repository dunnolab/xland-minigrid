# TODO: this is rendering mostly ported or adapted from the original Minigrid. A bit dirty right now...
import functools
import math
from typing import Optional, Union

import numpy as np

from ..core.constants import Colors, Tiles
from ..types import AgentState, IntOrArray
from .utils import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_hexagon,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

COLORS_MAP = {
    Colors.EMPTY: np.array((255, 255, 255)),  # just a placeholder
    Colors.RED: np.array((255, 0, 0)),
    Colors.GREEN: np.array((0, 255, 0)),
    Colors.BLUE: np.array((0, 0, 255)),
    Colors.PURPLE: np.array((112, 39, 195)),
    Colors.YELLOW: np.array((255, 255, 0)),
    Colors.GREY: np.array((100, 100, 100)),
    Colors.BLACK: np.array((0, 0, 0)),
    Colors.ORANGE: np.array((255, 140, 0)),
    Colors.WHITE: np.array((255, 255, 255)),
    Colors.BROWN: np.array((160, 82, 45)),
    Colors.PINK: np.array((225, 20, 147)),
}


def _render_empty(img: np.ndarray, color: int):
    fill_coords(img, point_in_rect(0.45, 0.55, 0.2, 0.65), COLORS_MAP[Colors.RED])
    fill_coords(img, point_in_rect(0.45, 0.55, 0.7, 0.85), COLORS_MAP[Colors.RED])

    fill_coords(img, point_in_rect(0, 0.031, 0, 1), COLORS_MAP[Colors.RED])
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), COLORS_MAP[Colors.RED])
    fill_coords(img, point_in_rect(1 - 0.031, 1, 0, 1), COLORS_MAP[Colors.RED])
    fill_coords(img, point_in_rect(0, 1, 1 - 0.031, 1), COLORS_MAP[Colors.RED])


def _render_floor(img: np.ndarray, color: int):
    # draw the grid lines (top and left edges)
    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
    # draw tile
    fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), COLORS_MAP[color] / 2)

    # # other grid lines (was used for paper visualizations)
    # fill_coords(img, point_in_rect(1 - 0.031, 1, 0, 1), (100, 100, 100))
    # fill_coords(img, point_in_rect(0, 1, 1 - 0.031, 1), (100, 100, 100))
    #


def _render_wall(img: np.ndarray, color: int):
    fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS_MAP[color])


def _render_ball(img: np.ndarray, color: int):
    _render_floor(img, Colors.BLACK)
    fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS_MAP[color])


def _render_square(img: np.ndarray, color: int):
    _render_floor(img, Colors.BLACK)
    fill_coords(img, point_in_rect(0.25, 0.75, 0.25, 0.75), COLORS_MAP[color])


def _render_pyramid(img: np.ndarray, color: int):
    _render_floor(img, Colors.BLACK)
    tri_fn = point_in_triangle(
        (0.15, 0.8),
        (0.85, 0.8),
        (0.5, 0.2),
    )
    fill_coords(img, tri_fn, COLORS_MAP[color])


def _render_hex(img: np.ndarray, color: int):
    _render_floor(img, Colors.BLACK)
    fill_coords(img, point_in_hexagon(0.35), COLORS_MAP[color])


def _render_star(img: np.ndarray, color: int):
    # yes, this is a hexagram not a star, but who cares...
    _render_floor(img, Colors.BLACK)
    tri_fn2 = point_in_triangle(
        (0.15, 0.75),
        (0.85, 0.75),
        (0.5, 0.15),
    )
    tri_fn1 = point_in_triangle(
        (0.15, 0.3),
        (0.85, 0.3),
        (0.5, 0.9),
    )
    fill_coords(img, tri_fn1, COLORS_MAP[color])
    fill_coords(img, tri_fn2, COLORS_MAP[color])


def _render_goal(img: np.ndarray, color: int):
    # draw the grid lines (top and left edges)
    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
    # draw tile
    fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), COLORS_MAP[color])

    # # other grid lines (was used for paper visualizations)
    # fill_coords(img, point_in_rect(1 - 0.031, 1, 0, 1), (100, 100, 100))
    # fill_coords(img, point_in_rect(0, 1, 1 - 0.031, 1), (100, 100, 100))


def _render_key(img: np.ndarray, color: int):
    _render_floor(img, Colors.BLACK)
    # Vertical quad
    fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), COLORS_MAP[color])
    # Teeth
    fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), COLORS_MAP[color])
    fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), COLORS_MAP[color])
    # Ring
    fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), COLORS_MAP[color])
    fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


def _render_door_locked(img: np.ndarray, color: int):
    fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), COLORS_MAP[color])
    fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * COLORS_MAP[color])
    # Draw key slot
    fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), COLORS_MAP[color])


def _render_door_closed(img: np.ndarray, color: int):
    fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), COLORS_MAP[color])
    fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
    fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), COLORS_MAP[color])
    fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))
    # Draw door handle
    fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), COLORS_MAP[color])


def _render_door_open(img: np.ndarray, color: int):
    _render_floor(img, Colors.BLACK)
    # draw the grid lines (top and left edges)
    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
    # draw door
    fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), COLORS_MAP[color])
    fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))


def _render_player(img: np.ndarray, direction: int):
    tri_fn = point_in_triangle(
        (0.12, 0.19),
        (0.87, 0.50),
        (0.12, 0.81),
    )
    # Rotate the agent based on its direction
    tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * (direction - 1))
    fill_coords(img, tri_fn, COLORS_MAP[Colors.RED])


TILES_FN_MAP = {
    Tiles.FLOOR: _render_floor,
    Tiles.WALL: _render_wall,
    Tiles.BALL: _render_ball,
    Tiles.SQUARE: _render_square,
    Tiles.PYRAMID: _render_pyramid,
    Tiles.HEX: _render_hex,
    Tiles.STAR: _render_star,
    Tiles.GOAL: _render_goal,
    Tiles.KEY: _render_key,
    Tiles.DOOR_LOCKED: _render_door_locked,
    Tiles.DOOR_CLOSED: _render_door_closed,
    Tiles.DOOR_OPEN: _render_door_open,
    Tiles.EMPTY: _render_empty,
}


# TODO: add highlight for can_see_through_walls=Fasle
def get_highlight_mask(grid: np.ndarray, agent: Optional[AgentState], view_size: int) -> np.ndarray:
    mask = np.zeros((grid.shape[0] + 2 * view_size, grid.shape[1] + 2 * view_size), dtype=np.bool_)
    if agent is None:
        return mask

    agent_y, agent_x = agent.position + view_size
    if agent.direction == 0:
        y, x = agent_y - view_size + 1, agent_x - (view_size // 2)
    elif agent.direction == 1:
        y, x = agent_y - (view_size // 2), agent_x
    elif agent.direction == 2:
        y, x = agent_y, agent_x - (view_size // 2)
    elif agent.direction == 3:
        y, x = agent_y - (view_size // 2), agent_x - view_size + 1
    else:
        raise RuntimeError("Unknown direction")

    mask[y : y + view_size, x : x + view_size] = True
    mask = mask[view_size:-view_size, view_size:-view_size]
    assert mask.shape == (grid.shape[0], grid.shape[1])

    return mask


@functools.cache
def render_tile(
    tile: tuple, agent_direction: Optional[int] = None, highlight: bool = False, tile_size: int = 32, subdivs: int = 3
) -> np.ndarray:
    img = np.full((tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8, fill_value=255)
    # draw tile
    TILES_FN_MAP[tile[0]](img, tile[1])
    # draw agent if on this tile
    if agent_direction is not None:
        _render_player(img, agent_direction)

    if highlight:
        highlight_img(img, alpha=0.2)

    # downsample the image to perform supersampling/anti-aliasing
    img = downsample(img, subdivs)

    return img


# WARN: will NOT work under jit and needed for debugging/presentation mainly.
def render(
    grid: np.ndarray,
    agent: Optional[AgentState] = None,
    view_size: IntOrArray = 7,
    tile_size: IntOrArray = 32,
) -> np.ndarray:
    # grid = np.asarray(grid)
    # compute the total grid size
    height_px = grid.shape[0] * int(tile_size)
    width_px = grid.shape[1] * int(tile_size)

    img = np.full((height_px, width_px, 3), dtype=np.uint8, fill_value=255)

    # compute agent fov highlighting
    highlight_mask = get_highlight_mask(grid, agent, int(view_size))

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if agent is not None and np.array_equal((y, x), agent.position):
                agent_direction = int(agent.direction)
            else:
                agent_direction = None

            tile_img = render_tile(
                tile=tuple(grid[y, x].tolist()),
                agent_direction=agent_direction,
                highlight=highlight_mask[y, x],
                tile_size=int(tile_size),
            )

            ymin = y * int(tile_size)
            ymax = (y + 1) * int(tile_size)
            xmin = x * int(tile_size)
            xmax = (x + 1) * int(tile_size)
            img[ymin:ymax, xmin:xmax, :] = tile_img

    return img
