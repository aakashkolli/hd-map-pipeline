import json

import numpy as np

from scripts.prepare_nuscenes_training import prepare_nuscenes_training
from src.pipeline.bev import BEVConfig


def test_prepare_nuscenes_training_aligns_image_and_mask_pixels(tmp_path):
    nuscenes_root = tmp_path / "nuscenes_mini"
    lidar_dir = nuscenes_root / "samples" / "LIDAR_TOP"
    maps_dir = nuscenes_root / "maps"
    lidar_dir.mkdir(parents=True)
    maps_dir.mkdir()

    lidar = np.array([[1.0, 1.0, 0.0, 5.0, 0.0]], dtype=np.float32)
    lidar.tofile(lidar_dir / "scene-0001.bin")
    (maps_dir / "scene-0001_annotations.json").write_text(
        json.dumps(
            {
                "lane_dividers": [
                    {"id": "lane-1", "geometry": [[1.0, 1.0, 0.0]]}
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "processed"

    result = prepare_nuscenes_training(
        nuscenes_root=nuscenes_root,
        output_dir=output_dir,
        bev_config=BEVConfig(resolution=0.5, extent=5.0),
    )

    assert result.pair_count == 1
    image = np.load(result.image_paths[0])
    mask = np.load(result.mask_paths[0])
    image_pixels = np.argwhere(image > 0.0)
    mask_pixels = np.argwhere(mask > 0)

    assert image_pixels.shape[0] == 1
    assert mask_pixels.shape[0] == 1
    np.testing.assert_array_equal(
        image_pixels[0],
        mask_pixels[0],
        err_msg="BEV image and annotation mask pixels are not aligned.",
    )


# ANTI-VIBE GATE - prepare_nuscenes_training.py
#
# 1. COORDINATE FRAME CONTRACT
#    nuScenes LiDAR points and map annotations are interpreted in the same
#    local world frame for this preparation step. BEV origin/resolution from
#    BEVConfig maps both image points and masks to identical pixels.
#
# 2. SILENT FAILURE MODE
#    If LiDAR and map annotations use different ego-pose origins, the script
#    can still write image/mask pairs that are spatially offset. Alignment is
#    checked by comparing known synthetic pixels.
#
# 3. VECTORIZATION STRATEGY
#    BEV image projection delegates to vectorized project_to_bev. Mask
#    rasterization converts all annotation vertices to pixel arrays and writes
#    them through NumPy indexing.
#
# 4. KNOWN LIMITATIONS
#    The mask rasterizes annotation vertices, not thick lane polygons. Real
#    training should draw anti-aliased line segments or buffered polylines.
#
# 5. OBSERVABILITY CHECK
#    In BEV visualization, lane mask pixels should sit directly over bright
#    LiDAR paint returns. Offset masks indicate origin, resolution, or frame
#    mismatch.
