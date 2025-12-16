from __future__ import annotations

import argparse
import os
from typing import Optional

import imageio.v3 as iio
import jax
import numpy as np
import pygame
import pygame.freetype
from pygame.event import Event

import xminigrid

from .environment import Environment, EnvParamsT
from .rendering.text_render import print_ruleset
from .types import EnvCarryT


class ManualControl:
    def __init__(
        self,
        env: Environment[EnvParamsT, EnvCarryT],
        env_params: EnvParamsT,
        agent_view: bool = False,
        save_video: bool = False,
        video_path: Optional[str] = None,
        video_format: str = ".mp4",
        video_fps: int = 8,
    ):
        self.env = env
        self.env_params = env_params
        self.agent_view = agent_view
        self.save_video = save_video
        self.video_path = video_path
        self.video_format = video_format
        self.video_fps = video_fps

        if self.save_video:
            self.frames = []

        self._reset = jax.jit(self.env.reset)
        self._step = jax.jit(self.env.step)
        self._key = jax.random.key(0)

        self.timestep = None

        self.render_size = None
        self.window = None
        self.clock = None
        self.screen_size = 640
        self.closed = False

    def render(self) -> None:
        assert self.timestep is not None

        if self.agent_view:
            img = self.timestep.observation
        else:
            img = self.env.render(self.env_params, self.timestep)

        if self.save_video:
            self.frames.append(img)

        # [h, w, c] -> [w, h, c]
        img = np.transpose(img, axes=(1, 0, 2))

        if self.render_size is None:
            self.render_size = img.shape[:2]

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("xland-minigrid")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(img)

        # Create background with mission description
        offset = surf.get_size()[0] * 0.1
        # offset = 32 if self.agent_pov else 64
        bg = pygame.Surface((int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset)))
        bg.convert()
        bg.fill((255, 255, 255))
        bg.blit(surf, (offset / 2, 0))

        bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

        font_size = 16
        text = "TODO: game goal encoding"
        font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
        text_rect = font.get_rect(text, size=font_size)
        text_rect.center = bg.get_rect().center
        text_rect.y = int(bg.get_height() - font_size * 1.5)
        font.render_to(bg, text_rect, text, size=font_size)

        self.window.blit(bg, (0, 0))
        pygame.event.pump()
        self.clock.tick(10)
        pygame.display.flip()

    def start(self) -> None:
        self.reset()

        """Start the window display with blocking event loop"""
        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: int) -> None:
        self.timestep = self._step(self.env_params, self.timestep, action)
        print(
            f"Step: {self.timestep.state.step_num} | ",
            f"StepType: {self.timestep.step_type} | ",
            f"Discount: {self.timestep.discount} | ",
            f"Reward: {self.timestep.reward}",
        )
        self.render()

        if self.timestep.last():
            self.reset()

    def reset(self) -> None:
        print("Reset!")
        self._key, reset_key = jax.random.split(self._key)

        self.timestep = self._reset(self.env_params, reset_key)
        self.render()
        print(
            f"Step: {self.timestep.state.step_num} |",
            f"StepType: {self.timestep.step_type} |",
            f"Discount: {self.timestep.discount} |",
            f"Reward: {self.timestep.reward}",
        )

    def key_handler(self, event: Event) -> None:
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.close()
            return

        key_to_action = {
            "up": 0,
            "right": 1,
            "left": 2,
            "tab": 3,
            "left shift": 4,
            "space": 5,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)

    def close(self) -> None:
        if self.window:
            pygame.quit()

        if self.save_video:
            assert self.video_path is not None
            save_path = os.path.join(self.video_path, f"manual_control_rollout{self.video_format}")
            if self.video_format == ".mp4":
                iio.imwrite(save_path, self.frames, format_hint=".mp4", fps=self.video_fps)
            elif self.video_format == ".gif":
                iio.imwrite(
                    save_path, self.frames[:-1], format_hint=".gif", duration=(1000 * 1 / self.video_fps), loop=10
                )
                # iio.imwrite(save_path, self.frames, format_hint=".gif", duration=(1000 * 1 / self.video_fps), loop=10)
            else:
                raise RuntimeError("Unknown video format! Should be one of ('.mp4', '.gif')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-5x5", choices=xminigrid.registered_environments())
    parser.add_argument("--benchmark-id", type=str, default="trivial-1m", choices=xminigrid.registered_benchmarks())
    parser.add_argument("--ruleset-id", type=int, default=0)
    parser.add_argument("--agent-view", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-path", type=str, default=".")
    parser.add_argument("--video-format", type=str, default=".mp4", choices=(".mp4", ".gif"))
    parser.add_argument("--video-fps", type=int, default=5)

    args = parser.parse_args()
    env, env_params = xminigrid.make(args.env_id)

    if args.agent_view:
        from xminigrid.experimental.img_obs import RGBImgObservationWrapper

        env = RGBImgObservationWrapper(env)

    if "XLand" in args.env_id:
        bench = xminigrid.load_benchmark(args.benchmark_id)
        ruleset = bench.get_ruleset(args.ruleset_id)

        env_params = env_params.replace(ruleset=ruleset)
        print_ruleset(ruleset)
        print()

    control = ManualControl(
        env=env,
        env_params=env_params,
        agent_view=args.agent_view,
        save_video=args.save_video,
        video_path=args.video_path,
        video_format=args.video_format,
        video_fps=args.video_fps,
    )
    control.start()
