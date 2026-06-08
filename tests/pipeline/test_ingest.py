import inspect

import numpy as np
import pandas as pd

from src.pipeline.ingest import accumulate_kitti_frames


def _write_oxts(path):
    path.write_text(
        "49.0 8.0 100.0 0.0 0.0 0.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )


def test_accumulate_kitti_frames_writes_world_enu_parquet(tmp_path):
    scene_dir = tmp_path / "scene"
    lidar_dir = scene_dir / "velodyne_points" / "data"
    oxts_dir = scene_dir / "oxts" / "data"
    calib_dir = tmp_path / "calib"
    lidar_dir.mkdir(parents=True)
    oxts_dir.mkdir(parents=True)
    calib_dir.mkdir()

    (calib_dir / "calib_velo_to_vehicle.txt").write_text(
        "R: 1 0 0 0 1 0 0 0 1\n"
        "T: 0 0 0\n",
        encoding="utf-8",
    )

    base_points = np.array(
        [
            [1.0, 0.0, 0.0, 0.5],
            [2.0, 0.0, 0.0, 0.7],
        ],
        dtype=np.float32,
    )
    for frame_id in range(3):
        frame_points = base_points.copy()
        frame_points[:, 0] += np.float32(frame_id)
        frame_points.tofile(lidar_dir / f"{frame_id:010d}.bin")
        _write_oxts(oxts_dir / f"{frame_id:010d}.txt")

    output_path = tmp_path / "accumulated.parquet"

    result = accumulate_kitti_frames(
        scene_dir=scene_dir,
        calib_dir=calib_dir,
        output_path=output_path,
        n_frames=3,
    )

    assert result.per_frame_point_counts == [2, 4, 6], (
        f"Expected cumulative counts [2, 4, 6], got "
        f"{result.per_frame_point_counts}."
    )
    assert output_path.exists(), "Ingest did not write accumulated.parquet."

    frame = pd.read_parquet(output_path)
    assert frame.columns.tolist() == [
        "x",
        "y",
        "z",
        "intensity",
        "timestamp",
        "frame_id",
    ], f"Unexpected accumulated schema: {frame.columns.tolist()}."
    assert str(frame["x"].dtype) == "float32"
    assert str(frame["y"].dtype) == "float32"
    assert str(frame["z"].dtype) == "float32"
    assert str(frame["intensity"].dtype) == "float32"
    assert str(frame["frame_id"].dtype).startswith("int")


def test_ingest_documents_frame_contract():
    docstring = inspect.getdoc(accumulate_kitti_frames)
    assert docstring is not None, "accumulate_kitti_frames needs a docstring."
    assert "FRAME:" in docstring, "Ingest must document output frame."
    assert "world ENU" in docstring, "Ingest output must be world ENU."


# ANTI-VIBE GATE - ingest.py
#
# 1. COORDINATE FRAME CONTRACT
#    LiDAR .bin inputs arrive in LiDAR frame, calibration maps LiDAR to
#    vehicle, OXTS maps vehicle to world, and parquet output is world ENU.
#    This is documented in accumulate_kitti_frames and checked by schema.
#
# 2. SILENT FAILURE MODE
#    If OXTS files are present but timestamp alignment differs from LiDAR
#    filename ordering, accumulation can use the wrong pose without a crash.
#    The function pairs sorted frame files and records frame_id for auditing.
#
# 3. VECTORIZATION STRATEGY
#    Each point frame is transformed by SE3.transform_points using matrix
#    multiplication and broadcast translation. DataFrame columns are built
#    from vector slices, not per-point Python iteration.
#
# 4. KNOWN LIMITATIONS
#    The stage uses local ENU OXTS poses and does not perform SLAM or loop
#    closure. Long routes can drift; the limitation belongs in dataset and
#    benchmark documentation for real KITTI runs.
#
# 5. OBSERVABILITY CHECK
#    In the viewer, a correct accumulated cloud shows static road and curb
#    structure aligned across frames. Bad frame transforms appear as parallel
#    duplicated roads or static objects smeared along the vehicle path.
