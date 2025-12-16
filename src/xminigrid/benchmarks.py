from __future__ import annotations

import bz2
import os
import pickle
import urllib.request
from typing import Callable, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import struct
from tqdm.auto import tqdm

from .types import RuleSet

HF_REPO_ID = os.environ.get("XLAND_MINIGRID_HF_REPO_ID", "Howuhh/xland_minigrid")
DATA_PATH = os.environ.get("XLAND_MINIGRID_DATA", os.path.expanduser("~/.xland_minigrid"))

NAME2HFFILENAME = {
    "trivial-21k": "trivial_21k",
    "small-1m": "small_1m_v3",
    "medium-1m": "medium_1m_v3",
    "medium-3m": "medium_3m_v3",
    "high-1m": "high_1m_v3",
    "high-3m": "high_3m_v3",
}


# jit compatible sampling and indexing!
# You can implement your custom curriculums based on this class.
class Benchmark(struct.PyTreeNode):
    goals: jax.Array
    rules: jax.Array
    init_tiles: jax.Array
    num_rules: jax.Array

    def num_rulesets(self) -> int:
        return len(self.goals)

    def get_ruleset(self, ruleset_id: Union[int, jax.Array]) -> RuleSet:
        return get_ruleset(self.goals, self.rules, self.init_tiles, ruleset_id)

    def sample_ruleset(self, key: jax.Array) -> RuleSet:
        ruleset_id = jax.random.randint(key, shape=(), minval=0, maxval=self.num_rulesets())
        return self.get_ruleset(ruleset_id)

    def shuffle(self, key: jax.Array) -> Benchmark:
        idxs = jax.random.permutation(key, jnp.arange(len(self.num_rules)))
        return jtu.tree_map(lambda a: a[idxs], self)

    def split(self, prop: float) -> tuple[Benchmark, Benchmark]:
        idx = round(len(self.num_rules) * prop)
        bench1 = jtu.tree_map(lambda a: a[:idx], self)
        bench2 = jtu.tree_map(lambda a: a[idx:], self)
        return bench1, bench2

    def filter_split(self, fn: Callable[[jax.Array, jax.Array], bool]) -> tuple[Benchmark, Benchmark]:
        # fn(single_goal, single_rules) -> bool
        mask = jax.vmap(fn)(self.goals, self.rules)
        bench1 = jtu.tree_map(lambda a: a[mask], self)
        bench2 = jtu.tree_map(lambda a: a[~mask], self)
        return bench1, bench2


def load_benchmark_from_path(path: str) -> Benchmark:
    benchmark_dict = load_bz2_pickle(path)
    benchmark = Benchmark(
        goals=benchmark_dict["goals"],
        rules=benchmark_dict["rules"],
        init_tiles=benchmark_dict["init_tiles"],
        num_rules=benchmark_dict["num_rules"],
    )
    return benchmark


def load_benchmark(name: str) -> Benchmark:
    if name not in NAME2HFFILENAME:
        raise RuntimeError(f"Unknown benchmark. Registered: {registered_benchmarks()}")

    os.makedirs(DATA_PATH, exist_ok=True)

    path = os.path.join(DATA_PATH, NAME2HFFILENAME[name])
    if not os.path.exists(path):
        _download_from_hf(HF_REPO_ID, NAME2HFFILENAME[name])

    return load_benchmark_from_path(path)


def registered_benchmarks() -> tuple[str, ...]:
    return tuple(NAME2HFFILENAME.keys())


def _download_from_hf(repo_id: str, filename: str) -> None:
    dataset_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"

    save_path = os.path.join(DATA_PATH, filename)
    print(f"Downloading benchmark data: {dataset_url} to {DATA_PATH}")

    with tqdm(unit="B", unit_scale=True, miniters=1, desc="Progress") as t:

        def progress_hook(block_num=1, block_size=1, total_size=None):
            if total_size is not None:
                t.total = total_size
            t.update(block_num * block_size - t.n)

        urllib.request.urlretrieve(dataset_url, save_path, reporthook=progress_hook)

    if not os.path.exists(os.path.join(DATA_PATH, filename)):
        raise IOError(f"Failed to download benchmark data from {dataset_url}")


def get_ruleset(
    goals: jax.Array,
    rules: jax.Array,
    init_tiles: jax.Array,
    ruleset_id: Union[int, jax.Array],
) -> RuleSet:
    goal = jax.lax.dynamic_index_in_dim(goals, ruleset_id, keepdims=False)
    rules = jax.lax.dynamic_index_in_dim(rules, ruleset_id, keepdims=False)
    init_tiles = jax.lax.dynamic_index_in_dim(init_tiles, ruleset_id, keepdims=False)

    return RuleSet(goal=goal, rules=rules, init_tiles=init_tiles)


def save_bz2_pickle(ruleset: dict[str, jax.Array], path: str, protocol: int = -1) -> None:
    with bz2.open(path, "wb") as f:
        pickle.dump(ruleset, f, protocol=protocol)


def load_bz2_pickle(path: str) -> dict[str, jax.Array]:
    with bz2.open(path, "rb") as f:
        ruleset = pickle.load(f)
    return ruleset
