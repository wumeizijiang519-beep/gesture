import os
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip("mediapipe")
from gesture.vision import Detector, verify_model

pytestmark = pytest.mark.vision


def test_official_model_load_and_blank_frame():
    model = Path(os.environ.get("GESTURE_MODEL", "models/hand_landmarker.task"))
    if not model.exists():
        pytest.skip("Download official model to run inference smoke test")
    assert len(verify_model(model)) == 64
    detector = Detector(model)
    try:
        assert detector.detect(np.zeros((480, 640, 3), np.uint8), 0., 0) is None
        assert detector.detect(np.zeros((480, 640, 3), np.uint8), .033, 1) is None
    finally:
        detector.close()
