# Code ported from https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/utils/rendering.py
from __future__ import annotations

import math
from typing import Callable, Union

import numpy as np
from typing_extensions import TypeAlias

Color: TypeAlias = Union[tuple[int, int, int], int, np.ndarray]
Point: TypeAlias = tuple[float, float]  # | np.ndarray


def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    img = img.reshape([img.shape[0] // factor, factor, img.shape[1] // factor, factor, 3])
    img = img.mean(axis=3)
    img = img.mean(axis=1)

    return img


def fill_coords(img: np.ndarray, fn: Callable, color: Color) -> np.ndarray:
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color

    return img


def rotate_fn(fin: Callable, cx: float, cy: float, theta: float) -> Callable:
    def fout(x, y):
        x = x - cx
        y = y - cy

        x2 = cx + x * math.cos(-theta) - y * math.sin(-theta)
        y2 = cy + y * math.cos(-theta) + x * math.sin(-theta)

        return fin(x2, y2)

    return fout


def point_in_line(x0: float, y0: float, x1: float, y1: float, r: float) -> Callable:
    p0 = np.array([x0, y0], dtype=np.float32)
    p1 = np.array([x1, y1], dtype=np.float32)
    dir = p1 - p0
    dist = np.linalg.norm(dir)
    dir = dir / dist

    xmin = min(x0, x1) - r
    xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r
    ymax = max(y0, y1) + r

    def fn(x, y):
        # Fast, early escape test
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

        q = np.array([x, y])
        pq = q - p0

        # Closest point on line
        a = np.dot(pq, dir)
        a = np.clip(a, 0, dist)
        p = p0 + a * dir

        dist_to_line = np.linalg.norm(q - p)
        return dist_to_line <= r

    return fn


def point_in_circle(cx: float, cy: float, r: float) -> Callable:
    def fn(x, y):
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r

    return fn


def point_in_rect(xmin: float, xmax: float, ymin: float, ymax: float) -> Callable:
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax

    return fn


def point_in_triangle(a: Point, b: Point, c: Point) -> Callable:
    a_ = np.array(a, dtype=np.float32)
    b_ = np.array(b, dtype=np.float32)
    c_ = np.array(c, dtype=np.float32)

    def fn(x, y):
        v0 = c_ - a_
        v1 = b_ - a_
        v2 = np.array((x, y)) - a_

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn


def point_in_hexagon(s: float) -> Callable:
    def fn(x, y):
        x = abs(x - 0.5)
        y = abs(y - 0.5)
        return y < 3**0.5 * min(s - x, s / 2)

    return fn


def highlight_img(img: np.ndarray, color: Color = (255, 255, 255), alpha: float = 0.30) -> None:
    blend_img = img + alpha * (np.array(color, dtype=np.uint8) - img)
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img[:, :, :] = blend_img
