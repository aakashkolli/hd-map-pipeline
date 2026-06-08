from pathlib import Path


def test_feature_renderer_uses_buffer_geometry_lines():
    source = Path("src/viz/src/renderer/FeatureRenderer.ts").read_text(
        encoding="utf-8"
    )

    assert "new THREE.BufferGeometry()" in source
    assert "new THREE.Line(" in source
    assert "Float32Array" in source
    assert "THREE.Geometry" not in source


def test_feature_renderer_exposes_rendered_feature_count():
    source = Path("src/viz/src/renderer/FeatureRenderer.ts").read_text(
        encoding="utf-8"
    )

    assert "renderedFeatureCount" in source
    assert "features.length" in source


# ANTI-VIBE GATE - FeatureRenderer.ts
#
# 1. COORDINATE FRAME CONTRACT
#    GeoJSON LineString coordinates are interpreted as world ENU xyz values
#    and uploaded directly to Three.js line buffers.
#
# 2. SILENT FAILURE MODE
#    Pixel coordinates can still render as lines, but they will be far from
#    the point cloud or compressed near the origin. QA should inspect overlay
#    alignment against lane paint.
#
# 3. VECTORIZATION STRATEGY
#    Feature positions are packed into Float32Array buffers before upload.
#    No Vector3 object is allocated per vertex.
#
# 4. KNOWN LIMITATIONS
#    Static tests verify renderer structure but not visual visibility. Browser
#    testing is still needed for line thickness and color contrast.
#
# 5. OBSERVABILITY CHECK
#    Loading 10 GeoJSON features should create 10 line objects in the scene and
#    each line should overlay the corresponding world-frame lane geometry.
