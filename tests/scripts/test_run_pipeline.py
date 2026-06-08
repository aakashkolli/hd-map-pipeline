import json

from scripts.run_pipeline import run_pipeline


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

