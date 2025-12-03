import argparse
import time
import timeit
from typing import Optional

import jax
import jax.tree_util as jtu
import numpy as np

import xminigrid
from xminigrid import load_benchmark
from xminigrid.wrappers import GymAutoResetWrapper

jax.config.update("jax_threefry_partitionable", True)

parser = argparse.ArgumentParser()
parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-16x16")
parser.add_argument("--benchmark-id", type=str, default="trivial-21k")
parser.add_argument("--img-obs", action="store_true")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--num-envs", type=int, default=8192)
parser.add_argument("--num-repeat", type=int, default=10, help="Number of timing repeats")
parser.add_argument("--num-iter", type=int, default=1, help="Number of runs during one repeat (time is summed)")


def build_benchmark(
    env_id: str,
    num_envs: int,
    timesteps: int,
    benchmark_id: Optional[str] = None,
    img_obs: bool = False,
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
    t = time.time()
    benchmark_fn().state.grid.block_until_ready()
    print(f"Compilation time: {time.time() - t}")
    times = timeit.repeat(
        lambda: benchmark_fn().state.grid.block_until_ready(),
        number=args.num_iter,
        repeat=args.num_repeat,
    )
    times = np.array(times) / args.num_iter

    elapsed_time = np.max(times)
    print(f"Elapsed time: {elapsed_time:.5f}s")
    return elapsed_time


if __name__ == "__main__":
    num_devices = jax.local_device_count()
    args = parser.parse_args()
    assert args.num_envs % num_devices == 0
    print("Num devices for pmap:", num_devices)

    # building for single env benchmarking
    benchmark_fn_single = build_benchmark(args.env_id, 1, args.timesteps, args.benchmark_id, args.img_obs)
    benchmark_fn_single = jax.jit(benchmark_fn_single)
    # building vmap for vectorization benchmarking
    benchmark_fn_vmap = build_benchmark(args.env_id, args.num_envs, args.timesteps, args.benchmark_id, args.img_obs)
    benchmark_fn_vmap = jax.jit(benchmark_fn_vmap)
    # building pmap for multi-gpu benchmarking (each doing (num_envs / num_devices) vmaps)
    benchmark_fn_pmap = build_benchmark(
        args.env_id, args.num_envs // num_devices, args.timesteps, args.benchmark_id, args.img_obs
    )
    benchmark_fn_pmap = jax.pmap(benchmark_fn_pmap)

    key = jax.random.key(0)
    pmap_keys = jax.random.split(key, num=num_devices)

    # benchmarking
    elapsed_time = timeit_benchmark(args, jtu.Partial(benchmark_fn_single, key))
    single_fps = args.timesteps / elapsed_time
    print(f"Single env, Elapsed time: {elapsed_time:.5f}s, FPS: {single_fps:.0f}")
    print()
    elapsed_time = timeit_benchmark(args, jtu.Partial(benchmark_fn_vmap, key))
    vmap_fps = (args.timesteps * args.num_envs) / elapsed_time
    print(f"Vmap env, Elapsed time: {elapsed_time:.5f}s, FPS: {vmap_fps:.0f}")
    print()
    elapsed_time = timeit_benchmark(args, jtu.Partial(benchmark_fn_pmap, pmap_keys))
    pmap_fps = (args.timesteps * args.num_envs) / elapsed_time
    print(f"Pmap env, Elapsed time: {elapsed_time:.5f}s, FPS: {pmap_fps:.0f}")
    print()
    print(f"FPS increase with vmap: {vmap_fps / single_fps:.0f}")
    print(f"FPS increase with pmap: {pmap_fps / single_fps:.0f}")
