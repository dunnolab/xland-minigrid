from .benchmarks import load_benchmark, registered_benchmarks
from .registration import make, register, registered_environments

# TODO: add __all__
__version__ = "0.9.2"

# ---------- XLand-MiniGrid environments ----------

# WARN: TMP, only for FPS measurements, will remove later
# register(
#     id="MiniGrid-1Rules",
#     entry_point="xminigrid.envs.xland_tmp:XLandMiniGrid",
#     num_rules=1,
#     height=16,
#     width=16,
# )
#
# register(
#     id="MiniGrid-3Rules",
#     entry_point="xminigrid.envs.xland_tmp:XLandMiniGrid",
#     num_rules=2,
#     height=16,
#     width=16,
# )
#
# register(
#     id="MiniGrid-6Rules",
#     entry_point="xminigrid.envs.xland_tmp:XLandMiniGrid",
#     num_rules=6,
#     height=16,
#     width=16,
# )
#
# register(
#     id="MiniGrid-12Rules",
#     entry_point="xminigrid.envs.xland_tmp:XLandMiniGrid",
#     num_rules=12,
#     height=16,
#     width=16,
# )
#
# register(
#     id="MiniGrid-24Rules",
#     entry_point="xminigrid.envs.xland_tmp:XLandMiniGrid",
#     num_rules=24,
#     height=16,
#     width=16,
# )

# register(
#     id="XLand-MiniGrid-R1-8x8",
#     entry_point="xminigrid.envs.xland:XLandMiniGrid",
#     grid_type="R1",
#     height=8,
#     width=8,
# )
#
# register(
#     id="XLand-MiniGrid-R1-16x16",
#     entry_point="xminigrid.envs.xland:XLandMiniGrid",
#     grid_type="R1",
#     height=16,
#     width=16,
# )
#
# register(
#     id="XLand-MiniGrid-R1-32x32",
#     entry_point="xminigrid.envs.xland:XLandMiniGrid",
#     grid_type="R1",
#     height=32,
#     width=32,
# )
#
# register(
#     id="XLand-MiniGrid-R1-64x64",
#     entry_point="xminigrid.envs.xland:XLandMiniGrid",
#     grid_type="R1",
#     height=64,
#     width=64,
# )

register(
    id="XLand-MiniGrid-R1-9x9",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R1",
    height=9,
    width=9,
)

register(
    id="XLand-MiniGrid-R1-11x11",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R1",
    height=11,
    width=11,
)

register(
    id="XLand-MiniGrid-R1-13x13",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R1",
    height=13,
    width=13,
)

register(
    id="XLand-MiniGrid-R1-15x15",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R1",
    height=15,
    width=15,
)

register(
    id="XLand-MiniGrid-R1-17x17",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R1",
    height=17,
    width=17,
)


register(
    id="XLand-MiniGrid-R2-9x9",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R2",
    height=9,
    width=9,
)

register(
    id="XLand-MiniGrid-R2-11x11",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R2",
    height=11,
    width=11,
)

register(
    id="XLand-MiniGrid-R2-13x13",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R2",
    height=13,
    width=13,
)

register(
    id="XLand-MiniGrid-R2-15x15",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R2",
    height=15,
    width=15,
)

register(
    id="XLand-MiniGrid-R2-17x17",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R2",
    height=17,
    width=17,
)


register(
    id="XLand-MiniGrid-R4-9x9",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R4",
    height=9,
    width=9,
)

register(
    id="XLand-MiniGrid-R4-11x11",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R4",
    height=11,
    width=11,
)

register(
    id="XLand-MiniGrid-R4-13x13",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R4",
    height=13,
    width=13,
)

register(
    id="XLand-MiniGrid-R4-15x15",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R4",
    height=15,
    width=15,
)

register(
    id="XLand-MiniGrid-R4-17x17",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R4",
    height=17,
    width=17,
)


register(
    id="XLand-MiniGrid-R6-13x13",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R6",
    height=13,
    width=13,
)

register(
    id="XLand-MiniGrid-R6-17x17",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R6",
    height=17,
    width=17,
)

register(
    id="XLand-MiniGrid-R6-19x19",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R6",
    height=19,
    width=19,
)

# 16, 19, 25
register(
    id="XLand-MiniGrid-R9-16x16",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R9",
    height=16,
    width=16,
)

register(
    id="XLand-MiniGrid-R9-19x19",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R9",
    height=19,
    width=19,
)

register(
    id="XLand-MiniGrid-R9-25x25",
    entry_point="xminigrid.envs.xland:XLandMiniGrid",
    grid_type="R9",
    height=25,
    width=25,
)

# ---------- Environments ported from MiniGrid ----------

# BlockedUnlockPickUp
register(
    id="MiniGrid-BlockedUnlockPickUp",
    entry_point="xminigrid.envs.minigrid.blockedunlockpickup:BlockedUnlockPickUp",
)

# DoorKey
register(
    id="MiniGrid-DoorKey-5x5",
    entry_point="xminigrid.envs.minigrid.doorkey:DoorKey",
    height=5,
    width=5,
)

register(
    id="MiniGrid-DoorKey-6x6",
    entry_point="xminigrid.envs.minigrid.doorkey:DoorKey",
    height=6,
    width=6,
)

register(
    id="MiniGrid-DoorKey-8x8",
    entry_point="xminigrid.envs.minigrid.doorkey:DoorKey",
    height=8,
    width=8,
)

register(
    id="MiniGrid-DoorKey-16x16",
    entry_point="xminigrid.envs.minigrid.doorkey:DoorKey",
    height=16,
    width=16,
)

# Empty
register(
    id="MiniGrid-Empty-5x5",
    entry_point="xminigrid.envs.minigrid.empty:Empty",
    height=5,
    width=5,
)

register(
    id="MiniGrid-Empty-6x6",
    entry_point="xminigrid.envs.minigrid.empty:Empty",
    height=6,
    width=6,
)

register(
    id="MiniGrid-Empty-8x8",
    entry_point="xminigrid.envs.minigrid.empty:Empty",
    height=8,
    width=8,
)

register(
    id="MiniGrid-Empty-16x16",
    entry_point="xminigrid.envs.minigrid.empty:Empty",
    height=16,
    width=16,
)

# EmptyRandom
register(
    id="MiniGrid-EmptyRandom-5x5",
    entry_point="xminigrid.envs.minigrid.empty:EmptyRandom",
    height=5,
    width=5,
)

register(
    id="MiniGrid-EmptyRandom-6x6",
    entry_point="xminigrid.envs.minigrid.empty:EmptyRandom",
    height=6,
    width=6,
)

register(
    id="MiniGrid-EmptyRandom-8x8",
    entry_point="xminigrid.envs.minigrid.empty:EmptyRandom",
    height=8,
    width=8,
)

register(
    id="MiniGrid-EmptyRandom-16x16",
    entry_point="xminigrid.envs.minigrid.empty:EmptyRandom",
    height=16,
    width=16,
)

# FourRooms
register(
    id="MiniGrid-FourRooms",
    entry_point="xminigrid.envs.minigrid.fourrooms:FourRooms",
)

# LockedRoom
register(
    id="MiniGrid-LockedRoom",
    entry_point="xminigrid.envs.minigrid.lockedroom:LockedRoom",
)

# Memory
register(
    id="MiniGrid-MemoryS8",
    entry_point="xminigrid.envs.minigrid.memory:Memory",
    width=8,
)

register(
    id="MiniGrid-MemoryS16",
    entry_point="xminigrid.envs.minigrid.memory:Memory",
    width=16,
)

register(
    id="MiniGrid-MemoryS32",
    entry_point="xminigrid.envs.minigrid.memory:Memory",
    width=32,
)

register(
    id="MiniGrid-MemoryS64",
    entry_point="xminigrid.envs.minigrid.memory:Memory",
    width=64,
)

register(
    id="MiniGrid-MemoryS128",
    entry_point="xminigrid.envs.minigrid.memory:Memory",
    width=128,
)

# PlayGround
register(
    id="MiniGrid-Playground",
    entry_point="xminigrid.envs.minigrid.playground:Playground",
)

# Unlock
register(
    id="MiniGrid-Unlock",
    entry_point="xminigrid.envs.minigrid.unlock:Unlock",
)

# UnlockPickUp
register(
    id="MiniGrid-UnlockPickUp",
    entry_point="xminigrid.envs.minigrid.unlockpickup:UnlockPickUp",
)
