import inspect
import json

import numpy as np

from src.data.nuscenes import (
    annotation_intersects_extent,
    load_map_annotations,
    load_nuscenes_lidar,
    load_nuscenes_scene,
)


def test_load_nuscenes_scene_reads_lidar_and_visible_annotations(tmp_path):
    root = tmp_path / "nuscenes_mini"
    lidar_dir = root / "samples" / "LIDAR_TOP"
    maps_dir = root / "maps"
    lidar_dir.mkdir(parents=True)
    maps_dir.mkdir()

    lidar = np.array(
        [[0.0, 0.0, 0.0, 0.5, 1.0], [1.0, 1.0, 0.0, 0.8, 2.0]],
        dtype=np.float32,
    )
    lidar.tofile(lidar_dir / "scene-0001.bin")
    annotation_path = maps_dir / "scene-0001_annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "lane_dividers": [
                    {
                        "id": "lane-1",
                        "geometry": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    scene = load_nuscenes_scene(root, scene_id="scene-0001")

    assert scene.lidar_points.shape == (2, 5)
    assert scene.lidar_points.dtype == np.float32
    assert len(scene.annotations) == 1
    assert annotation_intersects_extent(scene.annotations[0], extent=20.0)


def test_load_map_annotations_returns_world_frame_features(tmp_path):
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "lane_dividers": [
                    {
                        "id": "lane-1",
                        "geometry": [[2.0, 3.0, 0.0], [4.0, 5.0, 0.0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    annotations = load_map_annotations(annotation_path)

    assert annotations[0].geometry == [[2.0, 3.0, 0.0], [4.0, 5.0, 0.0]]
    assert annotations[0].source == "nuscenes_map"


def test_nuscenes_functions_document_frame_contracts():
    for function in (
        load_nuscenes_lidar,
        load_map_annotations,
        load_nuscenes_scene,
    ):
        docstring = inspect.getdoc(function)
        assert docstring is not None, f"{function.__name__} needs a docstring."
        assert "FRAME:" in docstring, (
            f"{function.__name__} must document spatial frame behavior."
        )


# ANTI-VIBE GATE - nuscenes.py
#
# 1. COORDINATE FRAME CONTRACT
#    LiDAR points are loaded in nuScenes sensor frame. Map annotations are
#    stored as local world-frame xyz polylines and returned unchanged as
#    LaneBoundaryFeature geometries.
#
# 2. SILENT FAILURE MODE
#    A valid annotation JSON can use a different local map origin than the
#    LiDAR sample. The parser preserves coordinates but cannot align frames
#    without calibrated ego pose metadata.
#
# 3. VECTORIZATION STRATEGY
#    LiDAR binary parsing uses np.fromfile and reshape. Annotation extent
#    checks convert geometry to one NumPy array and use vectorized bounds.
#
# 4. KNOWN LIMITATIONS
#    This parser covers the lightweight mini-scene representation used for
#    tests and preprocessing, not the full nuScenes database schema.
#
# 5. OBSERVABILITY CHECK
#    In BEV, parsed lane annotations should lie inside the road extent and
#    overlay the corresponding LiDAR ground points. Offset annotations signal
#    map-origin or ego-pose mismatch.
