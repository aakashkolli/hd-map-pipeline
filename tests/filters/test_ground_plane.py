import inspect

import numpy as np

from src.filters.ground_plane import RansacConfig, ransac_ground_plane


def test_ransac_ground_plane_finds_synthetic_plane_with_noise():
    rng = np.random.default_rng(42)
    ground = rng.uniform(-20.0, 20.0, (50_000, 3)).astype(np.float32)
    ground[:, 2] = rng.normal(0.0, 0.03, 50_000).astype(np.float32)
    noise = rng.uniform(-10.0, 10.0, (5_000, 3)).astype(np.float32)
    noise[:, 2] = rng.uniform(0.5, 3.0, 5_000).astype(np.float32)
    points = np.vstack([ground, noise]).astype(np.float32)
    cfg = RansacConfig(
        max_iterations=120,
        distance_threshold=0.15,
        min_inlier_ratio=0.35,
        seed_z_percentile=25.0,
        seed_xy_radius=30.0,
        random_seed=42,
    )

    result = ransac_ground_plane(points, cfg)

    assert result.inlier_ratio > 0.90, (
        f"Expected ground inlier ratio > 0.90, got {result.inlier_ratio:.3f}."
    )
    assert np.mean(result.ground_mask[: ground.shape[0]]) > 0.98, (
        "RANSAC missed too many synthetic ground points."
    )


def test_seed_filtered_ransac_finds_ground_when_full_cloud_fits_wall():
    rng = np.random.default_rng(8)
    ground = rng.uniform(-5.0, 5.0, (1_000, 3)).astype(np.float32)
    ground[:, 2] = rng.normal(0.0, 0.01, 1_000).astype(np.float32)
    wall = rng.uniform(-5.0, 5.0, (5_000, 3)).astype(np.float32)
    wall[:, 0] = rng.normal(2.0, 0.01, 5_000).astype(np.float32)
    wall[:, 2] = rng.uniform(1.0, 5.0, 5_000).astype(np.float32)
    points = np.vstack([ground, wall]).astype(np.float32)
    cfg = RansacConfig(
        max_iterations=100,
        distance_threshold=0.08,
        min_inlier_ratio=0.10,
        seed_z_percentile=10.0,
        seed_xy_radius=10.0,
        random_seed=7,
    )

    naive_plane = _naive_full_cloud_plane(points, cfg)
    result = ransac_ground_plane(points, cfg)

    assert abs(naive_plane[2]) < 0.50, (
        f"Naive full-cloud RANSAC should fit the wall, got normal {naive_plane[:3]}."
    )
    assert abs(result.plane[2]) > 0.95, (
        f"Seed-filtered RANSAC should fit ground, got normal {result.plane[:3]}."
    )
    assert np.mean(result.ground_mask[: ground.shape[0]]) > 0.95, (
        "Seed-filtered RANSAC did not retain the ground plane."
    )


def test_ground_plane_documents_frame_contract():
    docstring = inspect.getdoc(ransac_ground_plane)
    assert docstring is not None, "ransac_ground_plane needs a docstring."
    assert "FRAME:" in docstring, "Ground RANSAC must document input frame."
    assert "vehicle" in docstring, "Ground RANSAC must require vehicle frame."


def _naive_full_cloud_plane(points: np.ndarray, cfg: RansacConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.random_seed)
    best_plane = np.zeros(4, dtype=np.float64)
    best_count = 0
    points64 = points.astype(np.float64)

    for _ in range(cfg.max_iterations):
        sample_idx = rng.choice(points.shape[0], size=3, replace=False)
        sample = points64[sample_idx]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm <= np.finfo(np.float64).eps:
            continue
        normal = normal / norm
        d = -float(normal @ sample[0])
        distances = np.abs(points64 @ normal + d)
        count = int(np.count_nonzero(distances < cfg.distance_threshold))
        if count > best_count:
            best_count = count
            best_plane[:3] = normal
            best_plane[3] = d

    return best_plane


# ANTI-VIBE GATE - ground_plane.py
#
# 1. COORDINATE FRAME CONTRACT
#    Input points arrive in vehicle frame, where z is up relative to the
#    vehicle and seed thresholds are meaningful. Output masks index the same
#    vehicle-frame input points with no coordinate transform applied.
#
# 2. SILENT FAILURE MODE
#    A banked road or ramp can be fit as a plane, but nearby curb or obstacle
#    points may become inliers if the distance threshold is too permissive.
#    The inlier ratio and plane normal are diagnostics for this failure.
#
# 3. VECTORIZATION STRATEGY
#    Seed selection uses boolean masks over z and xy radius. Distance checks
#    compute points @ normal + d over the full cloud at once. No per-point
#    Python iteration is used.
#
# 4. KNOWN LIMITATIONS
#    The implementation fits one plane per frame and does not model terrain
#    curvature, overpasses, or multi-level roads. These limitations are
#    documented in the module docstring.
#
# 5. OBSERVABILITY CHECK
#    In the viewer, ground output should show the road surface and lane paint
#    without vertical walls. If wall faces remain in the ground layer, seed
#    filtering or the distance threshold is wrong.
