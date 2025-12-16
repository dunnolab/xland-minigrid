from typing import Optional, Union

import jax
import jax.numpy as jnp

from ..core.constants import Colors, Tiles
from ..types import AgentState, RuleSet

COLOR_NAMES = {
    Colors.EMPTY: "white",
    Colors.RED: "red",
    Colors.GREEN: "green",
    Colors.BLUE: "blue",
    Colors.PURPLE: "purple",
    Colors.YELLOW: "yellow",
    Colors.GREY: "grey",
    Colors.BLACK: "black",
    Colors.ORANGE: "orange",
    Colors.WHITE: "white",
    Colors.BROWN: "brown",
    Colors.PINK: "pink",
}

TILE_STR = {
    Tiles.EMPTY: " ",
    Tiles.FLOOR: ".",
    Tiles.WALL: "☰",
    Tiles.BALL: "⏺",
    Tiles.SQUARE: "▪",
    Tiles.PYRAMID: "▲",
    Tiles.HEX: "⬢",
    Tiles.STAR: "★",
    Tiles.GOAL: "■",
    Tiles.DOOR_LOCKED: "L",
    Tiles.DOOR_CLOSED: "C",
    Tiles.DOOR_OPEN: "O",
    Tiles.KEY: "K",
}

# for ruleset printing
RULE_TILE_STR = {
    Tiles.FLOOR: "floor",
    Tiles.BALL: "ball",
    Tiles.SQUARE: "square",
    Tiles.PYRAMID: "pyramid",
    Tiles.GOAL: "goal",
    Tiles.KEY: "key",
    Tiles.HEX: "hexagon",
    Tiles.STAR: "star",
}

PLAYER_STR = {0: "^", 1: ">", 2: "V", 3: "<"}


def _wrap_with_color(string: str, color: str) -> str:
    return f"[bold {color}]{string}[/bold {color}]"


# WARN: will NOT work under jit and needed for debugging mainly.
def render(grid: jax.Array, agent: Optional[AgentState] = None) -> str:
    string = ""

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            tile_id, tile_color = grid[y, x]
            tile_str = TILE_STR[tile_id.item()]
            tile_color = COLOR_NAMES[tile_color.item()]

            if agent is not None and jnp.all(agent.position == jnp.array((y, x))):
                tile_str = PLAYER_STR[agent.direction.item()]
                tile_color = COLOR_NAMES[Colors.RED]

            string += _wrap_with_color(tile_str, tile_color)

        if y < grid.shape[0] - 1:
            string += "\n"

    return string


# WARN: This is for debugging mainly! Will refactor later if needed.
def _encode_tile(tile: list[int]) -> str:
    return f"{COLOR_NAMES[tile[1]]} {RULE_TILE_STR[tile[0]]}"


def _text_encode_goal(goal: list[int]) -> str:
    goal_id = goal[0]
    if goal_id == 1:
        return f"AgentHold({_encode_tile(goal[1:3])})"
    elif goal_id == 3:
        return f"AgentNear({_encode_tile(goal[1:3])})"
    elif goal_id == 4:
        return f"TileNear({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
    elif goal_id == 7:
        return f"TileNearUpGoal({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
    elif goal_id == 8:
        return f"TileNearRightGoal({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
    elif goal_id == 9:
        return f"TileNearDownGoal({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
    elif goal_id == 10:
        return f"TileNearLeftGoal({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
    elif goal_id == 11:
        return f"AgentNearUpGoal({_encode_tile(goal[1:3])})"
    elif goal_id == 12:
        return f"AgentNearRightGoal({_encode_tile(goal[1:3])})"
    elif goal_id == 13:
        return f"AgentNearDownGoal({_encode_tile(goal[1:3])})"
    elif goal_id == 14:
        return f"AgentNearLeftGoal({_encode_tile(goal[1:3])})"
    else:
        raise RuntimeError(f"Rendering: Unknown goal id: {goal_id}")


def _text_encode_rule(rule: list[int]) -> str:
    rule_id = rule[0]
    if rule_id == 1:
        return f"AgentHold({_encode_tile(rule[1:3])}) -> {_encode_tile(rule[3:5])}"
    elif rule_id == 2:
        return f"AgentNear({_encode_tile(rule[1:3])}) -> {_encode_tile(rule[3:5])}"
    elif rule_id == 3:
        return f"TileNear({_encode_tile(rule[1:3])}, {_encode_tile(rule[3:5])}) -> {_encode_tile(rule[5:7])}"
    elif rule_id == 4:
        return f"TileNearUpRule({_encode_tile(rule[1:3])}, {_encode_tile(rule[3:5])}) -> {_encode_tile(rule[5:7])}"
    elif rule_id == 5:
        return f"TileNearRightRule({_encode_tile(rule[1:3])}, {_encode_tile(rule[3:5])}) -> {_encode_tile(rule[5:7])}"
    elif rule_id == 6:
        return f"TileNearDownRule({_encode_tile(rule[1:3])}, {_encode_tile(rule[3:5])}) -> {_encode_tile(rule[5:7])}"
    elif rule_id == 7:
        return f"TileNearLeftRule({_encode_tile(rule[1:3])}, {_encode_tile(rule[3:5])}) -> {_encode_tile(rule[5:7])}"
    elif rule_id == 8:
        return f"AgentNearUpRule({_encode_tile(rule[1:3])}) -> {_encode_tile(rule[3:5])}"
    elif rule_id == 9:
        return f"AgentNearRightRule({_encode_tile(rule[1:3])}) -> {_encode_tile(rule[3:5])}"
    elif rule_id == 10:
        return f"AgentNearDownRule({_encode_tile(rule[1:3])}) -> {_encode_tile(rule[3:5])}"
    elif rule_id == 11:
        return f"AgentNearLeftRule({_encode_tile(rule[1:3])}) -> {_encode_tile(rule[3:5])}"
    else:
        raise RuntimeError(f"Rendering: Unknown rule id: {rule_id}")


def print_ruleset(ruleset: RuleSet) -> None:
    print("GOAL:")
    print(_text_encode_goal(ruleset.goal.tolist()))
    print()
    print("RULES:")
    for rule in ruleset.rules.tolist():
        if rule[0] != 0:
            print(_text_encode_rule(rule))
    print()
    print("INIT TILES:")
    for tile in ruleset.init_tiles.tolist():
        if tile[0] != 0:
            print(_encode_tile(tile))
