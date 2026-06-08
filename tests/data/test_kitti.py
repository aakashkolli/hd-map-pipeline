import inspect

import numpy as np

from src.data.kitti import (
    OxtsOrigin,
    load_lidar_frame,
    parse_calibration_file,
    parse_oxts_pose,
)
from src.geometry.transforms import SE3


def test_load_lidar_frame_reads_kitti_bin_shape_and_dtype(tmp_path):
    points = np.array(
        [
            [1.0, 2.0, 3.0, 0.5],
            [4.0, 5.0, 6.0, 0.9],
        ],
        dtype=np.float32,
    )
    bin_path = tmp_path / "0000000000.bin"
    points.tofile(bin_path)

    loaded = load_lidar_frame(bin_path)

    assert loaded.shape == (2, 4), (
        f"Expected KITTI LiDAR shape (2, 4), got {loaded.shape}."
    )
    assert loaded.dtype == np.float32, (
        f"Expected float32 LiDAR points, got {loaded.dtype}."
    )
    np.testing.assert_allclose(
        loaded,
        points,
        err_msg="KITTI .bin reader changed LiDAR point values.",
    )


def test_parse_calibration_file_returns_labeled_se3(tmp_path):
    calib_path = tmp_path / "calib_velo_to_vehicle.txt"
    calib_path.write_text(
        "R: 1 0 0 0 1 0 0 0 1\n"
        "T: 1.5 -0.25 0.75\n",
        encoding="utf-8",
    )

    transform = parse_calibration_file(
        calib_path,
        source_frame="lidar",
        target_frame="vehicle",
    )

    assert isinstance(transform, SE3), "Calibration parser must return SE3."
    assert transform.source_frame == "lidar"
    assert transform.target_frame == "vehicle"
    np.testing.assert_allclose(
        transform.translation,
        np.array([1.5, -0.25, 0.75]),
        err_msg="Parsed calibration translation does not match file values.",
    )


def test_parse_oxts_pose_returns_world_vehicle_se3(tmp_path):
    oxts_path = tmp_path / "0000000000.txt"
    oxts_path.write_text(
        "49.0 8.0 100.0 0.0 0.0 0.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )
    origin = OxtsOrigin(latitude=49.0, longitude=8.0, altitude=100.0)

    pose = parse_oxts_pose(oxts_path, origin=origin)

    assert isinstance(pose, SE3), "OXTS pose parser must return SE3."
    assert pose.source_frame == "vehicle"
    assert pose.target_frame == "world"
    np.testing.assert_allclose(
        pose.translation,
        np.zeros(3),
        atol=1e-5,
        err_msg="Pose at the origin should have zero ENU translation.",
    )


def test_kitti_spatial_functions_document_frame_contracts():
    for function in (load_lidar_frame, parse_calibration_file, parse_oxts_pose):
        docstring = inspect.getdoc(function)
        assert docstring is not None, f"{function.__name__} needs a docstring."
        assert "FRAME:" in docstring, (
            f"{function.__name__} must document spatial frame semantics."
        )


# ANTI-VIBE GATE - kitti.py
#
# 1. COORDINATE FRAME CONTRACT
#    load_lidar_frame returns raw KITTI Velodyne points in LiDAR frame.
#    parse_calibration_file returns an SE3 from caller-specified source to
#    target frame. parse_oxts_pose returns vehicle-to-world ENU pose. Each
#    contract is documented in the function docstring and verified here.
#
# 2. SILENT FAILURE MODE
#    A calibration file can be syntactically valid but semantically mislabeled
#    by the caller, such as passing target_frame="vehicle" for a camera
#    calibration. SE3 labels make this visible, but the parser cannot infer
#    sensor intent from arbitrary filenames.
#
# 3. VECTORIZATION STRATEGY
#    LiDAR .bin parsing uses np.fromfile followed by reshape, calibration
#    values use array reshape, and OXTS translation uses vector arithmetic.
#    No Python iteration over point arrays is used.
#
# 4. KNOWN LIMITATIONS
#    OXTS conversion uses a local ENU approximation around the first origin.
#    This is acceptable for short KITTI segments, but long survey routes
#    should use a geodesy library or map projection with datum handling.
#
# 5. OBSERVABILITY CHECK
#    Correct output in the viewer shows accumulated static objects aligned
#    across frames in ENU. Bad OXTS or calibration parsing appears as doubled
#    lane markings, tilted roads, or frame-to-frame drift of buildings.
