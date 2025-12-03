# This is not the fastest implementation, but c'mon,
# I only have to run it once in forever...
# Meanwhile, make yourself a cup of tea and relax, tqdm go brrr...
# P.S. If you are willing to improve this, submit a PR! Beware that generation should remain deterministic!
import argparse
import random
from itertools import product

import jax.numpy as jnp
from tqdm.auto import tqdm, trange

from xminigrid.benchmarks import save_bz2_pickle
from xminigrid.core.constants import Colors, Tiles
from xminigrid.core.goals import (
    AgentHoldGoal,
    AgentNearDownGoal,
    AgentNearGoal,
    AgentNearLeftGoal,
    AgentNearRightGoal,
    AgentNearUpGoal,
    TileNearDownGoal,
    TileNearGoal,
    TileNearLeftGoal,
    TileNearRightGoal,
    TileNearUpGoal,
)
from xminigrid.core.grid import pad_along_axis
from xminigrid.core.rules import (
    AgentHoldRule,
    AgentNearDownRule,
    AgentNearLeftRule,
    AgentNearRightRule,
    AgentNearRule,
    AgentNearUpRule,
    EmptyRule,
    TileNearDownRule,
    TileNearLeftRule,
    TileNearRightRule,
    TileNearRule,
    TileNearUpRule,
)

COLORS = [
    Colors.RED,
    Colors.GREEN,
    Colors.BLUE,
    Colors.PURPLE,
    Colors.YELLOW,
    Colors.GREY,
    Colors.WHITE,
    Colors.BROWN,
    Colors.PINK,
    Colors.ORANGE,
]

# we need to distinguish between them, to avoid sampling
# near(goal, goal) goal or rule as goal tiles are not pickable
NEAR_TILES_LHS = list(
    product([Tiles.BALL, Tiles.SQUARE, Tiles.PYRAMID, Tiles.KEY, Tiles.STAR, Tiles.HEX, Tiles.GOAL], COLORS)
)
# these are pickable!
NEAR_TILES_RHS = list(product([Tiles.BALL, Tiles.SQUARE, Tiles.PYRAMID, Tiles.KEY, Tiles.STAR, Tiles.HEX], COLORS))

HOLD_TILES = list(product([Tiles.BALL, Tiles.SQUARE, Tiles.PYRAMID, Tiles.KEY, Tiles.STAR, Tiles.HEX], COLORS))

# to imitate disappearance production rule
PROD_TILES = list(product([Tiles.BALL, Tiles.SQUARE, Tiles.PYRAMID, Tiles.KEY, Tiles.STAR, Tiles.HEX], COLORS))
PROD_TILES = PROD_TILES + [(Tiles.FLOOR, Colors.BLACK)]


def encode(ruleset):
    flatten_encoding = jnp.concatenate([ruleset["goal"].encode(), *[r.encode() for r in ruleset["rules"]]]).tolist()
    return tuple(flatten_encoding)


def diff(list1, list2):
    return list(set(list1) - set(list2))


def sample_goal():
    goals = (
        AgentHoldGoal,
        # agent near variations
        AgentNearGoal,
        AgentNearUpGoal,
        AgentNearDownGoal,
        AgentNearLeftGoal,
        AgentNearRightGoal,
        # tile near variations
        TileNearGoal,
        TileNearUpGoal,
        TileNearDownGoal,
        TileNearLeftGoal,
        TileNearRightGoal,
    )
    goal_idx = random.randint(0, 10)
    if goal_idx == 0:
        tile = random.choice(HOLD_TILES)
        goal = goals[0](tile=jnp.array(tile))
        return goal, (tile,)
    elif 1 <= goal_idx <= 5:
        tile = random.choice(NEAR_TILES_LHS)
        goal = goals[goal_idx](tile=jnp.array(tile))
        return goal, (tile,)
    elif 6 <= goal_idx <= 10:
        tile_a = random.choice(NEAR_TILES_LHS)
        tile_b = random.choice(NEAR_TILES_RHS)
        goal = goals[goal_idx](tile_a=jnp.array(tile_a), tile_b=jnp.array(tile_b))
        return goal, (tile_a, tile_b)
    else:
        raise RuntimeError("Unknown goal")


def sample_rule(prod_tile, used_tiles):
    rules = (
        AgentHoldRule,
        # agent near variations
        AgentNearRule,
        AgentNearUpRule,
        AgentNearDownRule,
        AgentNearLeftRule,
        AgentNearRightRule,
        # tile near variations
        TileNearRule,
        TileNearUpRule,
        TileNearDownRule,
        TileNearLeftRule,
        TileNearRightRule,
    )
    rule_idx = random.randint(0, 10)

    if rule_idx == 0:
        tile = random.choice(diff(HOLD_TILES, used_tiles))
        rule = rules[rule_idx](tile=jnp.array(tile), prod_tile=jnp.array(prod_tile))
        return rule, (tile,)
    elif 1 <= rule_idx <= 5:
        tile = random.choice(diff(HOLD_TILES, used_tiles))
        rule = rules[rule_idx](tile=jnp.array(tile), prod_tile=jnp.array(prod_tile))
        return rule, (tile,)
    elif 6 <= rule_idx <= 10:
        tile_a = random.choice(diff(NEAR_TILES_LHS, used_tiles))
        tile_b = random.choice(diff(NEAR_TILES_RHS, used_tiles))
        rule = rules[rule_idx](tile_a=jnp.array(tile_a), tile_b=jnp.array(tile_b), prod_tile=jnp.array(prod_tile))
        return rule, (tile_a, tile_b)
    else:
        raise RuntimeError("Unknown rule")


# See Appendix A.2 in "Human-timescale adaptation in an open-ended task space" for sampling procedure.
# We tried to follow this procedure closely here (for single-agent environments).
# There is two options: choose one branch or sample for all branches (like a full binary tree)
# We sample binary tree here (pruning it along the way).
def sample_ruleset(
    chain_depth: int,
    num_distractor_rules: int,
    num_distractor_objects: int,
    sample_depth: bool,
    sample_distractor_rules: bool,
    prune_chain: bool,
    # actually, we can vary prune_prob on each sample to diversify even further
    prune_prob: float = 0.0,
):
    used_tiles = []
    chain_tiles = []

    # sample goal first
    goal, goal_tiles = sample_goal()
    used_tiles.extend(goal_tiles)
    chain_tiles.extend(goal_tiles)

    # sample main rules in a chain
    rules = []
    init_tiles = []

    num_levels = random.randint(0, chain_depth) if sample_depth else chain_depth
    # there is no rules, just one goal, thus we need to add goal tiles to init tiles
    if num_levels == 0:
        # WARN: you really should add distractor objects in this case, as without them goal will be obvious
        init_tiles.extend(goal_tiles)
        # one empty rule as a placeholder, to fill up "rule" key, this will not introduce overhead under jit
        rules.append(EmptyRule())

    # for logging
    for level in range(num_levels):
        next_chain_tiles = []

        # sampling in a chain, rules inputs from previous layer are rule results from this layer
        while chain_tiles:
            prod_tile = chain_tiles.pop()
            if prune_chain and random.random() < prune_prob:
                # prune this branch and add this tile to initial tiles
                init_tiles.append(prod_tile)
                continue

            rule, rule_tiles = sample_rule(prod_tile, used_tiles)
            used_tiles.extend(rule_tiles)
            next_chain_tiles.extend(rule_tiles)
            rules.append(rule)

            # inputs to the last rules in the chain are the initial tiles for the current level
            if level == num_levels - 1:
                init_tiles.extend(rule_tiles)

        chain_tiles = next_chain_tiles

    # sample distractor objects
    init_tiles.extend(random.choices(diff(NEAR_TILES_LHS, used_tiles), k=num_distractor_objects))
    # sample distractor rules
    if sample_distractor_rules:
        num_distractor_rules = random.randint(0, num_distractor_rules)

    for _ in range(num_distractor_rules):
        prod_tile = random.choice(diff(PROD_TILES, used_tiles))
        # distractors can include already sampled tiles, to create dead-end rules
        rule, rule_tiles = sample_rule(prod_tile, used_tiles=[])
        rules.append(rule)
        init_tiles.extend(rule_tiles)

    # if for some reason there are no rules, add one empty (we will ignore it later)
    if len(rules) == 0:
        rules.append(EmptyRule())

    return {
        "goal": goal,
        "rules": rules,
        "init_tiles": init_tiles,
        # additional info (for example for biasing sampling by number of rules)
        # you can add other field if needed, just copy-paste this file!
        # saving counts, as later they will be padded to the same size
        "num_rules": len([r for r in rules if not isinstance(r, EmptyRule)]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain_depth", type=int, default=1)
    parser.add_argument("--sample_depth", action="store_true")
    parser.add_argument("--prune_chain", action="store_true")
    parser.add_argument("--prune_prob", type=float, default=0.5)
    parser.add_argument("--num_distractor_rules", type=int, default=0)
    parser.add_argument("--sample_distractor_rules", action="store_true")
    parser.add_argument("--num_distractor_objects", type=int, default=0)
    parser.add_argument("--total_rulesets", type=int, default=100_000)
    parser.add_argument("--save_path", type=str, default="./xland_generated_rulesets")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.random_seed)
    # we need keep track of unique already sampled rulesets
    unique_rulesets_encodings = set()
    rulesets = []
    for _ in trange(args.total_rulesets):
        ruleset = sample_ruleset(
            args.chain_depth,
            args.num_distractor_rules,
            args.num_distractor_objects,
            args.sample_depth,
            args.sample_distractor_rules,
            args.prune_chain,
            args.prune_prob,
        )
        # sample only unique rulesets (rejection sampling)
        if encode(ruleset) in unique_rulesets_encodings:
            tqdm.write("Collision, resampling...!")
            while encode(ruleset) in unique_rulesets_encodings:
                ruleset = sample_ruleset(
                    args.chain_depth,
                    args.num_distractor_rules,
                    args.num_distractor_objects,
                    args.sample_depth,
                    args.sample_distractor_rules,
                    args.prune_chain,
                    args.prune_prob,
                )

        rulesets.append(
            {
                "goal": ruleset["goal"].encode(),
                "rules": jnp.vstack([r.encode() for r in ruleset["rules"]]),
                "init_tiles": jnp.array(ruleset["init_tiles"], dtype=jnp.uint8),
                "num_rules": jnp.asarray(ruleset["num_rules"], dtype=jnp.uint8),
            }
        )
        unique_rulesets_encodings.add(encode(ruleset))

    del unique_rulesets_encodings
    # concatenating padded rulesets, for convenient sampling in jax
    # as in jax we can not retrieve single item from the list/pytree under jit
    # also all rulesets in one benchmark should have same shapes to work under jit
    max_rules = max(map(lambda r: len(r["rules"]), rulesets))
    max_tiles = max(map(lambda r: len(r["init_tiles"]), rulesets))
    print("Max rules:", max_rules)
    print("Max init tiles:", max_tiles)

    # goals:      [total_rulesets, goal_encoding_dim]
    # rules:      [total_rulesets, max_rules, rules_encoding_dim]
    # init_tiles: [total_rulesets, max_tiles, 2]
    print("Padding and concatenating...")
    concat_rulesets = {
        "generation_config": vars(args),
        "goals": jnp.vstack([r["goal"] for r in rulesets]),
        "rules": jnp.vstack([pad_along_axis(r["rules"], pad_to=max_rules)[None, ...] for r in rulesets]),
        "init_tiles": jnp.vstack([pad_along_axis(r["init_tiles"], pad_to=max_tiles)[None, ...] for r in rulesets]),
        "num_rules": jnp.vstack([r["num_rules"] for r in rulesets]),
    }
    print("Saving...")
    save_bz2_pickle(concat_rulesets, args.save_path, protocol=-1)

    # # for debugging only
    # save_bz2_pickle(rulesets, args.save_path + "_raw", protocol=-1)
