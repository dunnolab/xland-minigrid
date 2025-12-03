# Performance benchmark for all environments. For the paper and to check regressions after new features.
import argparse
import pprint
import timeit
from typing import Optional

import jax
import jax.tree_util as jtu
import numpy as np
from tqdm.auto import tqdm

import xminigrid
from xminigrid import load_benchmark
from xminigrid.wrappers import GymAutoResetWrapper

jax.config.update("jax_threefry_partitionable", True)

NUM_ENVS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark-id", type=str, default="trivial-21k")
parser.add_argument("--img-obs", action="store_true")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--num-repeat", type=int, default=10, help="Number of timing repeats")
parser.add_argument("--num-iter", type=int, default=1, help="Number of runs during one repeat (time is summed)")


def build_benchmark(
    env_id: str, num_envs: int, timesteps: int, benchmark_id: Optional[str] = None, img_obs: bool = False
):
    env, env_params = xminigrid.make(env_id)
    env = GymAutoResetWrapper(env)

    # enable img observations if needed
    if img_obs:
        from xminigrid.experimental.img_obs import RGBImgObservationWrapper

        env = RGBImgObservationWrapper(env)

    # choose XLand benchmark if needed
    if "XLand-MiniGrid" in env_id and benchmark_id is not None:
        ruleset = load_benchmark(benchmark_id).sample_ruleset(jax.random.key(0))
        env_params = env_params.replace(ruleset=ruleset)

    def benchmark_fn(key):
        def _body_fn(timestep, action):
            new_timestep = jax.vmap(env.step, in_axes=(None, 0, 0))(env_params, timestep, action)
            return new_timestep, None

        key, actions_key = jax.random.split(key)
        keys = jax.random.split(key, num=num_envs)
        actions = jax.random.randint(
            actions_key, shape=(timesteps, num_envs), minval=0, maxval=env.num_actions(env_params)
        )

        timestep = jax.vmap(env.reset, in_axes=(None, 0))(env_params, keys)
        # unroll can affect FPS greatly !!!
        timestep = jax.lax.scan(_body_fn, timestep, actions, unroll=1)[0]
        return timestep

    return benchmark_fn


# see https://stackoverflow.com/questions/56763416/what-is-diffrence-between-number-and-repeat-in-python-timeit
# on why we divide by args.num_iter
def timeit_benchmark(args, benchmark_fn):
    benchmark_fn().state.grid.block_until_ready()
    times = timeit.repeat(
        lambda: benchmark_fn().state.grid.block_until_ready(),
        number=args.num_iter,
        repeat=args.num_repeat,
    )
    times = np.array(times) / args.num_iter
    elapsed_time = np.max(times)
    return elapsed_time


# that can take a while!
if __name__ == "__main__":
    num_devices = jax.local_device_count()
    args = parser.parse_args()
    print("Num devices:", num_devices)

    environments = xminigrid.registered_environments()
    summary = {}
    for num_envs in tqdm(NUM_ENVS, desc="Benchmark", leave=False):
        results = {}
        for env_id in tqdm(environments, desc="Envs.."):
            assert num_envs % num_devices == 0
            # building pmap for multi-gpu benchmarking (each doing (num_envs / num_devices) vmaps)
            benchmark_fn_pmap = build_benchmark(
                env_id, num_envs // num_devices, args.timesteps, args.benchmark_id, args.img_obs
            )
            benchmark_fn_pmap = jax.pmap(benchmark_fn_pmap)

            # benchmarking
            pmap_keys = jax.random.split(jax.random.key(0), num=num_devices)

            elapsed_time = timeit_benchmark(args, jtu.Partial(benchmark_fn_pmap, pmap_keys))
            pmap_fps = (args.timesteps * num_envs) // elapsed_time

            results[env_id] = int(pmap_fps)
        summary[num_envs] = results

    pprint.pprint(summary)
