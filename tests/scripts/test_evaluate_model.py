import numpy as np

from scripts.evaluate_model import compute_class_iou


def test_compute_class_iou_reports_lane_line_metric():
    prediction = np.array([[0, 1], [0, 1]], dtype=np.int64)
    target = np.array([[0, 1], [1, 1]], dtype=np.int64)

    iou = compute_class_iou(prediction, target, num_classes=2)

    assert iou[1] > 0.35

