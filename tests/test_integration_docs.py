from pathlib import Path


def test_integration_files_exist_and_readme_is_standalone():
    required = [
        "docker/Dockerfile.pipeline",
        "docker/Dockerfile.viz",
        "docker/docker-compose.yml",
        "src/viz/package.json",
        "src/viz/index.html",
        "src/viz/vite.config.ts",
        "src/viz/tsconfig.json",
        "scripts/measure_viewer_fps.mjs",
        "docs/viewer_screenshot.png",
        "README.md",
        "docs/resume_bullets.md",
    ]
    for path in required:
        assert Path(path).exists(), f"Missing integration file: {path}"

    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Quick Start" in readme
    assert "```mermaid" in readme
    assert "docs/viewer_screenshot.png" in readme
    assert "500K" in readme and "60.66 FPS" in readme
    assert "Dataset Setup" in readme
    assert "Known Limitations" in readme
    assert "PRD.md" not in readme and "CLAUDE.md" not in readme


def test_resume_bullets_are_under_word_limit():
    lines = [
        line.strip()
        for line in Path("docs/resume_bullets.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("- ")
    ]

    assert len(lines) == 2
    for line in lines:
        assert len(line[2:].split()) <= 30, f"Resume bullet too long: {line}"


# ANTI-VIBE GATE - Docker integration and public documentation
#
# 1. COORDINATE FRAME CONTRACT
#    Docker integration does not transform spatial data directly. The
#    pipeline container runs scripts/run_pipeline.py, which preserves the
#    world ENU frame contracts documented in src/pipeline modules and writes
#    GeoJSON feature coordinates in that same map frame.
#
# 2. SILENT FAILURE MODE
#    A Docker image can build while excluding required runtime files from the
#    build context. test_integration_files_exist_and_readme_is_standalone
#    catches this by requiring Dockerfiles, the compose file, src/viz/index.html,
#    Vite config, and public documentation to exist together.
#
# 3. VECTORIZATION STRATEGY
#    This integration layer performs no point-array processing. Point-cloud
#    computation remains inside the tested numpy/C/PyTorch components, and the
#    container smoke path exercises those implementations without adding Python
#    iteration over point arrays.
#
# 4. KNOWN LIMITATIONS
#    The test validates file presence and standalone documentation, not live
#    Docker runtime behavior. Runtime behavior is validated separately with
#    docker compose up pipeline --build --abort-on-container-exit and a viz
#    container HTTP/build smoke check.
#
# 5. OBSERVABILITY CHECK
#    Correct output is visible as the Vite app shell titled "HD Map Pipeline
#    Viewer" and as data/outputs/features.geojson plus qa_report.json produced
#    by the pipeline container for viewer inspection.
