import inspect
from pathlib import Path

import numpy as np

from src.filters.voxel import voxel_downsample


def test_voxel_downsample_empty_input_returns_empty_float32():
    points = np.empty((0, 4), dtype=np.float32)

    result = voxel_downsample(points, voxel_size=0.5)

    assert result.shape == (0, 4), f"Expected empty (0, 4), got {result.shape}."
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}."


def test_voxel_downsample_keeps_one_point_per_voxel():
    points = np.array(
        [
            [0.10, 0.10, 0.10, 0.20],
            [0.20, 0.20, 0.20, 0.80],
            [1.10, 0.10, 0.10, 0.40],
            [0.10, 1.10, 0.10, 0.60],
        ],
        dtype=np.float32,
    )

    result = voxel_downsample(points, voxel_size=1.0)

    assert result.shape == (3, 4), (
        f"Expected one representative from each of 3 voxels, got {result.shape}."
    )
    voxel_indices = np.floor(result[:, :3] / np.float32(1.0)).astype(np.int64)
    unique_voxels = np.unique(voxel_indices, axis=0)
    assert unique_voxels.shape[0] == result.shape[0], (
        "Downsampled output contains duplicate voxel representatives."
    )


def test_voxel_downsample_documents_frame_contract():
    docstring = inspect.getdoc(voxel_downsample)
    assert docstring is not None, "voxel_downsample needs a docstring."
    assert "FRAME:" in docstring, "Voxel filter must document frame behavior."
    assert "unchanged" in docstring, "Voxel filter must preserve input frame."


def test_voxel_extension_files_exist():
    ext_root = Path(__file__).parents[2] / "src" / "ext"

    assert (ext_root / "voxel_filter.cpp").exists(), "Missing C++ voxel filter."
    assert (ext_root / "bindings.cpp").exists(), "Missing pybind11 bindings."
    assert (ext_root / "CMakeLists.txt").exists(), "Missing CMake build file."


# ANTI-VIBE GATE - voxel_filter.cpp + bindings
#
# 1. COORDINATE FRAME CONTRACT
#    Input points arrive in whatever spatial frame the caller provides, and
#    output points leave in that same unchanged frame. The filter only groups
#    coordinates into voxels; it does not rotate, translate, or relabel data.
#
# 2. SILENT FAILURE MODE
#    A voxel size chosen larger than lane-marking width can erase geometric
#    detail while still returning a valid point cloud. The mitigation is to
#    source voxel_size from config and inspect downsample density in the viewer.
#
# 3. VECTORIZATION STRATEGY
#    The public Python path delegates point-array iteration to the compiled
#    extension when available. The Python fallback uses np.unique over voxel
#    indices rather than per-point Python loops.
#
# 4. KNOWN LIMITATIONS
#    The representative point is the first point in each voxel, not a centroid.
#    This preserves intensity values but can bias geometry toward scan order.
#
# 5. OBSERVABILITY CHECK
#    In the viewer, downsampled clouds should retain road edges and lane paint
#    while reducing dense overlapping points. Over-large voxels make lane
#    markings look blocky or disappear.
