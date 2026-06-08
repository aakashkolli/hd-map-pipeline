import json
import struct

import numpy as np
import pytest

from scripts.run_pipeline import _write_points_bin, run_pipeline


def test_run_pipeline_full_smoke_outputs_features_and_qa(tmp_path):
    run_pipeline(
        config_path="configs/default.yaml",
        stage="full",
        output_dir=tmp_path,
    )

    features_path = tmp_path / "features.geojson"
    qa_path = tmp_path / "qa_report.json"
    assert features_path.exists(), "Full pipeline did not write features.geojson."
    assert qa_path.exists(), "Full pipeline did not write qa_report.json."

    features = json.loads(features_path.read_text(encoding="utf-8"))
    coords = features["features"][0]["geometry"]["coordinates"][0]
    assert coords[0] >= 0.0, "GeoJSON should contain world-frame coordinates."

    report = json.loads(qa_path.read_text(encoding="utf-8"))
    assert report["completeness"] > 0.0


def test_run_pipeline_full_smoke_writes_points_bin(tmp_path):
    run_pipeline(
        config_path="configs/default.yaml",
        stage="full",
        output_dir=tmp_path,
    )
    bin_path = tmp_path / "points.bin"
    assert bin_path.exists(), "Full pipeline did not write points.bin."

    data = bin_path.read_bytes()
    n = struct.unpack_from("<I", data, 0)[0]
    assert n > 0
    expected = 4 + n * 3 * 4 + n * 4
    assert len(data) == expected, f"Binary size mismatch: got {len(data)}, expected {expected}"


def test_write_points_bin_roundtrip(tmp_path):
    """_write_points_bin format must match DataLoader.ts wire contract."""
    rng = np.random.default_rng(0)
    n = 100
    points = rng.standard_normal((n, 4)).astype(np.float32)
    points[:, 3] = np.clip(points[:, 3], 0, 1)

    out = tmp_path / "test.bin"
    _write_points_bin(out, points)

    data = out.read_bytes()
    # Header: uint32 N little-endian
    (n_read,) = struct.unpack_from("<I", data, 0)
    assert n_read == n

    # XYZ block: N*3 float32 starting at byte 4
    xyz_bytes = data[4 : 4 + n * 12]
    xyz = np.frombuffer(xyz_bytes, dtype="<f4").reshape(n, 3)
    np.testing.assert_allclose(xyz, points[:, :3], atol=1e-6)

    # Intensity block: N float32 starting at byte 4 + N*12
    intensity_bytes = data[4 + n * 12 : 4 + n * 12 + n * 4]
    intensity = np.frombuffer(intensity_bytes, dtype="<f4")
    np.testing.assert_allclose(intensity, points[:, 3], atol=1e-6)

