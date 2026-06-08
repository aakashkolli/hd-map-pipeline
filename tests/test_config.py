from configs import load_config


def test_load_config_exposes_nested_attributes():
    cfg = load_config("configs/default.yaml")

    assert cfg.pipeline.world_frame == "enu"
    assert cfg.filters.ransac.distance_threshold == 0.15
    assert cfg.extraction.dbscan_min_samples == 10

