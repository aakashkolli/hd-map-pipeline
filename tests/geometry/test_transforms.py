import inspect

import numpy as np

from src.geometry.transforms import SE3


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.standard_normal((3, 3))
    q, _ = np.linalg.qr(matrix)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_se3_round_trip():
    """Applying a transform and its inverse recovers points within tolerance."""
    rng = np.random.default_rng(42)
    transform = SE3(
        rotation=_random_rotation(rng),
        translation=rng.standard_normal(3),
        source_frame="lidar",
        target_frame="world",
    )
    points = rng.standard_normal((1000, 3)).astype(np.float32)

    recovered = transform.inverse().transform_points(
        transform.transform_points(points)
    )

    np.testing.assert_allclose(
        points,
        recovered,
        atol=1e-5,
        err_msg="SE3 round-trip failed for lidar-to-world points.",
    )


def test_compose_matches_sequential_application():
    """Composed transforms match sequential frame conversion."""
    rng = np.random.default_rng(7)
    t_vehicle_lidar = SE3(
        rotation=_random_rotation(rng),
        translation=rng.standard_normal(3),
        source_frame="lidar",
        target_frame="vehicle",
    )
    t_world_vehicle = SE3(
        rotation=_random_rotation(rng),
        translation=rng.standard_normal(3),
        source_frame="vehicle",
        target_frame="world",
    )
    points_lidar = rng.standard_normal((1000, 3)).astype(np.float32)

    sequential = t_world_vehicle.transform_points(
        t_vehicle_lidar.transform_points(points_lidar)
    )
    composed = t_vehicle_lidar.compose(t_world_vehicle).transform_points(
        points_lidar
    )

    np.testing.assert_allclose(
        sequential,
        composed,
        atol=1e-5,
        err_msg="SE3 composition did not match sequential lidar-to-world transform.",
    )


def test_inverse_composition_is_identity():
    """A transform composed with its inverse leaves source-frame points unchanged."""
    rng = np.random.default_rng(11)
    transform = SE3(
        rotation=_random_rotation(rng),
        translation=rng.standard_normal(3),
        source_frame="vehicle",
        target_frame="world",
    )
    points_vehicle = rng.standard_normal((1000, 3)).astype(np.float32)

    identity = transform.compose(transform.inverse())
    recovered = identity.transform_points(points_vehicle)

    np.testing.assert_allclose(
        points_vehicle,
        recovered,
        atol=1e-5,
        err_msg="SE3 inverse composition failed to preserve vehicle-frame points.",
    )


def test_transform_points_documents_frame_contract():
    """Spatial point arrays must document source and target frame semantics."""
    docstring = inspect.getdoc(SE3.transform_points)

    assert docstring is not None, "transform_points must have a docstring."
    assert "FRAME:" in docstring, "Point array frame contract must be explicit."
    assert "SOURCE frame" in docstring, "Input frame semantics are missing."
    assert "TARGET frame" in docstring, "Output frame semantics are missing."


# ANTI-VIBE GATE - transforms.py
#
# 1. COORDINATE FRAME CONTRACT
#    Input points arrive in the transform's source frame and leave in
#    the transform's target frame. The contract is documented in the
#    SE3 class docstring and transform_points docstring via SOURCE,
#    TARGET, and FRAME annotations.
#
# 2. SILENT FAILURE MODE
#    A numerically orthogonal matrix with the wrong semantic frame labels
#    will pass algebraic checks while moving points into the wrong frame.
#    The test catches the algebra, while source_frame and target_frame
#    labels make frame intent inspectable at call sites.
#
# 3. VECTORIZATION STRATEGY
#    transform_points uses one matrix multiplication over points @ R.T
#    plus a broadcast translation. There is no Python iteration over
#    point arrays.
#
# 4. KNOWN LIMITATIONS
#    SE3 validates shape and rotation orthogonality, but it cannot prove
#    a calibration file used the intended sensor convention. Dataset
#    loaders must preserve source and target frame labels when parsing.
#
# 5. OBSERVABILITY CHECK
#    Correct output in the viewer preserves road geometry in world ENU:
#    flat roads remain level and sequential frame accumulation aligns
#    repeated static structures instead of duplicating or tilting them.
