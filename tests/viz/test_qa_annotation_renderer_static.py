from pathlib import Path


def test_qa_annotation_renderer_uses_specified_colors_and_buffers():
    source = Path("src/viz/src/renderer/QAAnnotationRenderer.ts").read_text(
        encoding="utf-8"
    )

    assert "0xf59e0b" in source
    assert "0xef4444" in source
    assert "new THREE.BufferGeometry()" in source
    assert "THREE.Geometry" not in source


def test_qa_annotation_renderer_has_click_metrics_callback():
    source = Path("src/viz/src/renderer/QAAnnotationRenderer.ts").read_text(
        encoding="utf-8"
    )

    assert "onAnnotationSelected" in source
    assert "userData" in source


# ANTI-VIBE GATE - QAAnnotationRenderer.ts
#
# 1. COORDINATE FRAME CONTRACT
#    False-positive and missed-GT geometries are world ENU polylines rendered
#    directly into the same scene as point clouds and feature lines.
#
# 2. SILENT FAILURE MODE
#    If QA IDs do not match feature IDs, colors render but sidebar metrics show
#    the wrong annotation. The renderer stores metadata in userData for click
#    handling.
#
# 3. VECTORIZATION STRATEGY
#    Annotation coordinates are packed into Float32Array buffers for Three.js
#    upload. No legacy per-vertex Geometry API is used.
#
# 4. KNOWN LIMITATIONS
#    Static tests verify color constants and callback wiring, but real click
#    selection still needs browser raycaster testing.
#
# 5. OBSERVABILITY CHECK
#    False positives should render amber and missed GT red. Clicking an
#    annotation should surface its QA metrics in the sidebar.
