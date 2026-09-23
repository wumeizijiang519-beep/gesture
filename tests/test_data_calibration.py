import json
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from gesture.data import Recorder, read_episode, verify_episode, export_npz
from gesture.geometry import Intrinsics
from gesture.calibration import make_board, solve_calibration
from gesture.vision import verify_model


def minimal_row(t=1 / 30):
    return {"sim_time": t, "capture_time": None, "frame_id": -1, "observation": None,
            "robot": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "joints": [0] * 9},
            "command": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "grip": .08, "active": False},
            "motor_targets": [0] * 9, "state": "DISARMED"}


def test_record_export_integrity_and_no_overwrite(tmp_path):
    rec = Recorder(tmp_path / "episode", {"source": "unit_test"})
    rec.append(minimal_row())
    rec.append(minimal_row(2 / 30))
    rec.close({"success": False})
    assert verify_episode(rec.path)
    meta, rows = read_episode(rec.path)
    assert meta["source"] == "unit_test" and len(rows) == 2
    export_npz(rec.path, tmp_path / "out.npz")
    with np.load(tmp_path / "out.npz", allow_pickle=False) as data:
        assert data["qpos"].shape == (2, 9)
        assert data["uv"].shape == (2, 21, 2)
        assert data["state"].dtype.kind == "U"
    with pytest.raises(FileExistsError):
        Recorder(rec.path, {})
    with pytest.raises(FileExistsError):
        export_npz(rec.path, tmp_path / "out.npz")
    (rec.path / "steps.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_episode(rec.path)


def test_corrupt_or_nonmonotonic_records_fail(tmp_path):
    rec = Recorder(tmp_path / "ep", {})
    rec.append(minimal_row())
    rec.append(minimal_row())
    rec.close()
    with pytest.raises(ValueError):
        read_episode(rec.path)
    with (rec.path / "steps.jsonl").open("w") as f:
        f.write("{broken}")
    with pytest.raises(ValueError):
        read_episode(rec.path)


def test_model_checksum_rejects_corruption(tmp_path):
    path = tmp_path / "fake.task"
    path.write_bytes(b"this is not a model")
    with pytest.raises(ValueError):
        verify_model(path)
    with pytest.raises(FileNotFoundError):
        verify_model(tmp_path / "missing.task")


def test_board_and_calibration_solver(tmp_path):
    make_board(tmp_path / "board.svg")
    text = (tmp_path / "board.svg").read_text()
    assert 'width="297mm"' in text and "20.0 mm" in text
    rng = np.random.default_rng(7)
    k = Intrinsics(640, 480, 520., 515., 320., 240.)
    board = np.zeros((54, 3), np.float32)
    board[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * .02
    views = []
    for _ in range(20):
        rv = Rotation.from_euler("xyz", rng.uniform(-.4, .4, 3)).as_rotvec()
        tv = np.array([-.08, -.05, .6]) + rng.uniform([-.03, -.03, -.1], [.03, .03, .1])
        points = cv2.projectPoints(board, rv, tv, k.matrix, np.zeros(5))[0]
        views.append(points)
    recovered, report = solve_calibration(views, (640, 480))
    assert recovered.calibrated
    assert report["rms_px"] < .001
    np.testing.assert_allclose(recovered.matrix, k.matrix, atol=.03)
    with pytest.raises(ValueError):
        solve_calibration(views[:2], (640, 480))


def test_manifest_rejects_traversal(tmp_path):
    rec = Recorder(tmp_path / "ep", {})
    rec.append(minimal_row())
    rec.close()
    manifest = json.loads((rec.path / "manifest.json").read_text())
    manifest["../outside"] = "fake"
    (rec.path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_episode(rec.path)
