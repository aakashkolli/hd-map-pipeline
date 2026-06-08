from pathlib import Path


def test_viewer_benchmark_mode_uses_configured_500k_point_count():
    source = Path("src/viz/src/main.ts").read_text(encoding="utf-8")
    config = Path("configs/viz.json").read_text(encoding="utf-8")

    assert "viewerConfig.syntheticPointCount" in source
    assert "generateSyntheticPointCloud" in source
    assert '"syntheticPointCount": 500000' in config
    assert "500000" not in source


def test_viewer_exposes_runtime_fps_metrics_for_headless_measurement():
    source = Path("src/viz/src/main.ts").read_text(encoding="utf-8")

    assert "__HD_MAP_VIEWER_METRICS__" in source
    assert "framesRendered" in source
    assert "averageFps" in source
    assert "benchmarkFrameCount" in source


def test_viewer_fps_benchmark_script_uses_chrome_devtools_metrics():
    source = Path("scripts/measure_viewer_fps.mjs").read_text(encoding="utf-8")

    assert "Browser.getVersion" in source
    assert "__HD_MAP_VIEWER_METRICS__" in source
    assert "averageFps" in source
    assert "process.exit(1)" in source


# ANTI-VIBE GATE - viewer benchmark mode
#
# 1. COORDINATE FRAME CONTRACT
#    Synthetic benchmark positions are generated directly in world ENU, then
#    loaded into PointCloudRenderer.load, whose docstring defines positions as
#    world ENU xyz triples. No LiDAR-to-world transform is applied here.
#
# 2. SILENT FAILURE MODE
#    A benchmark can report smooth FPS while rendering too few points. This
#    test requires main.ts to use viewerConfig.syntheticPointCount and requires
#    configs/viz.json to set that count to 500000.
#
# 3. VECTORIZATION STRATEGY
#    Browser rendering uses Float32Array buffers passed to BufferGeometry.
#    Python tests do static checks only and do not iterate over point arrays.
#
# 4. KNOWN LIMITATIONS
#    Static tests prove benchmark hooks exist, but FPS must still be measured
#    by running the Vite app in Chrome or the Docker viz service.
#
# 5. OBSERVABILITY CHECK
#    Correct output shows a dense synthetic world-frame cloud in the viewer and
#    window.__HD_MAP_VIEWER_METRICS__ reports framesRendered plus averageFps.
