from pathlib import Path


def test_point_cloud_renderer_uses_buffer_geometry_and_disposes():
    source = Path("src/viz/src/renderer/PointCloudRenderer.ts").read_text(
        encoding="utf-8"
    )

    assert "new THREE.BufferGeometry()" in source
    assert "Float32Array" in source
    assert "this.geometry.dispose()" in source
    assert "THREE.Geometry" not in source


def test_color_mode_updates_color_buffer_without_geometry_rebuild():
    source = Path("src/viz/src/renderer/PointCloudRenderer.ts").read_text(
        encoding="utf-8"
    )
    method = source.split("setColorMode", maxsplit=1)[1].split("dispose():", maxsplit=1)[0]

    assert "needsUpdate = true" in method
    assert "new THREE.BufferGeometry()" not in method
    assert "dispose()" not in method


# ANTI-VIBE GATE - PointCloudRenderer.ts
#
# 1. COORDINATE FRAME CONTRACT
#    The renderer receives positions that are already world ENU. It does not
#    transform vehicle or LiDAR frame points; camera controls only affect view.
#
# 2. SILENT FAILURE MODE
#    Passing vehicle-frame points still renders a cloud, but accumulated roads
#    appear tilted or duplicated. The pipeline ingest stage is responsible for
#    world-frame parquet output before visualization.
#
# 3. VECTORIZATION STRATEGY
#    Positions, intensities, and colors use Float32Array buffers and Three.js
#    BufferAttribute uploads. No per-point object allocation is used.
#
# 4. KNOWN LIMITATIONS
#    Static analysis verifies API usage but does not measure FPS directly.
#    Browser FPS is measured separately by scripts/measure_viewer_fps.mjs
#    against the configured 500K benchmark scene.
#
# 5. OBSERVABILITY CHECK
#    In the viewer, loading a second scene should not grow GPU memory
#    unbounded, and color mode changes should update instantly without cloud
#    flicker or geometry re-upload.
