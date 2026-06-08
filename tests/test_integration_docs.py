from pathlib import Path


def test_integration_files_exist_and_readme_is_standalone():
    required = [
        "docker/Dockerfile.pipeline",
        "docker/Dockerfile.viz",
        "docker/docker-compose.yml",
        "src/viz/package.json",
        "src/viz/vite.config.ts",
        "src/viz/tsconfig.json",
        "README.md",
        "docs/resume_bullets.md",
    ]
    for path in required:
        assert Path(path).exists(), f"Missing integration file: {path}"

    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Quick Start" in readme
    assert "```mermaid" in readme
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

