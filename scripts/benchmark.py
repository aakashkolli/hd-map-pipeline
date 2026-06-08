#!/usr/bin/env python3
"""Performance benchmarks for pipeline stages."""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ext._voxel_filter import voxel_downsample_cpp
from src.filters.voxel import _voxel_downsample_numpy


VOXEL_TARGET_MS = 80.0
VOXEL_INPUT_POINTS = 200_000
VOXEL_OUTPUT_POINTS = 20_000
VOXEL_SIZE_METERS = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage benchmarks.")
    parser.add_argument("--stage", required=True, choices=("voxel",))
    parser.add_argument("--n_iterations", type=int, default=10)
    args = parser.parse_args()

    if args.stage == "voxel":
        run_voxel_benchmark(args.n_iterations)


def run_voxel_benchmark(n_iterations: int) -> None:
    rng = np.random.default_rng(42)
    repeats_per_voxel = VOXEL_INPUT_POINTS // VOXEL_OUTPUT_POINTS
    voxel_ids = np.arange(VOXEL_OUTPUT_POINTS, dtype=np.float32)
    base_points = np.column_stack(
        [
            (voxel_ids % 200.0) * VOXEL_SIZE_METERS,
            ((voxel_ids // 200.0) % 100.0) * VOXEL_SIZE_METERS,
            (voxel_ids // 20_000.0) * VOXEL_SIZE_METERS,
            np.zeros(VOXEL_OUTPUT_POINTS, dtype=np.float32),
        ]
    ).astype(np.float32)
    points = np.repeat(base_points, repeats_per_voxel, axis=0)
    points[:, :3] += rng.uniform(
        0.0,
        VOXEL_SIZE_METERS * 0.25,
        (VOXEL_INPUT_POINTS, 3),
    ).astype(np.float32)
    points[:, 3] = rng.uniform(0.0, 1.0, VOXEL_INPUT_POINTS).astype(np.float32)

    cpp_timings = _time_function(
        lambda: voxel_downsample_cpp(points, VOXEL_SIZE_METERS),
        n_iterations=n_iterations,
    )
    numpy_timings = _time_function(
        lambda: _voxel_downsample_numpy(points, voxel_size=VOXEL_SIZE_METERS),
        n_iterations=n_iterations,
    )
    cpp_mean = statistics.mean(cpp_timings)
    numpy_mean = statistics.mean(numpy_timings)
    speedup = numpy_mean / cpp_mean

    print("Stage: voxel")
    print(f"Hardware: {platform.machine()} CPU, {platform.platform()}")
    print(f"Input: {VOXEL_INPUT_POINTS:,} points")
    print(_format_stats("C++", cpp_timings))
    print(_format_stats("NumPy", numpy_timings))
    print(f"Speedup: {speedup:.2f}x")
    print(f"Target: < {VOXEL_TARGET_MS:.0f}ms")
    print(f"Status: {'PASS' if cpp_mean < VOXEL_TARGET_MS else 'FAIL'}")


def _time_function(function, *, n_iterations: int) -> list[float]:
    timings = []
    function()
    for _ in range(n_iterations):
        start = time.perf_counter()
        function()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def _format_stats(name: str, timings: list[float]) -> str:
    return (
        f"{name} mean: {statistics.mean(timings):.2f}ms | "
        f"std: {statistics.pstdev(timings):.2f}ms | "
        f"min: {min(timings):.2f}ms | max: {max(timings):.2f}ms"
    )


if __name__ == "__main__":
    main()
