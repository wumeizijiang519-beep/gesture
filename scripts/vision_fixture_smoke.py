"""Opt-in public-image inference/PnP/CFR-video smoke test; NOT webcam performance validation.

Fixtures are downloaded from MediaPipe's official asset host, never committed or
redistributed. Their measured hashes are recorded. The model itself is hash-pinned.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import tempfile
import urllib.request
import cv2
from gesture.cli import parser as app_parser
from gesture.data import dump, read_episode
from gesture.geometry import Intrinsics, PoseEstimator
from gesture.runtime import run
from gesture.vision import Detector, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/hand_landmarker.task")
    parser.add_argument("--output", default="validation/public-image-smoke.json")
    args = parser.parse_args()
    results = []
    accepted_image = None
    with tempfile.TemporaryDirectory(prefix="gesture-public-fixtures-") as temp:
        root = Path(temp)
        for name in ("thumb_up.jpg", "pointing_up.jpg", "right_hands.jpg"):
            url = f"https://storage.googleapis.com/mediapipe-assets/{name}"
            path = root / name
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(10 * 1024 * 1024 + 1)
            if len(data) > 10 * 1024 * 1024:
                raise ValueError("Unexpectedly large public test fixture")
            path.write_bytes(data)
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot decode {name}")
            detector = Detector(args.model)
            try:
                obs = detector.detect(image, 0., 0)
                record = {"name": name, "url": url, "sha256": sha256(path),
                          "detector_reason": detector.reason}
                if name == "right_hands.jpg":
                    assert obs is None and detector.reason == "multiple_hands", record
                else:
                    assert obs is not None, record
                    estimator = PoseEstimator(Intrinsics.approximate(image.shape[1], image.shape[0]))
                    pose = estimator.estimate(obs)
                    record.update({"pnp_reason": estimator.reason,
                                   "reprojection_px": pose.reprojection_px if pose else None})
                    if pose is not None:
                        accepted_image = image.copy()
                results.append(record)
            finally:
                detector.close()
        assert accepted_image is not None, "No public single-hand fixture passed the PnP gates"
        h, w = accepted_image.shape[:2]
        # CFR video validates decode -> real neural inference -> geometry -> control -> simulation -> log.
        video = root / "static-hand.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30., (w, h))
        if not writer.isOpened():
            raise RuntimeError("MJPG encoder unavailable")
        for _ in range(45):
            writer.write(accepted_image)
        writer.release()
        out = root / "episode"
        config = app_parser().parse_args(["video", "--input", str(video), "--model", args.model,
                                         "--auto-arm", "--headless", "--seconds", "1",
                                         "--output", str(out)])
        run(config)
        _, steps = read_episode(out)
        active = sum(row["command"]["active"] for row in steps)
        assert active > 0, "The real-image video pipeline never engaged"
        result = {"kind": "public_static_image_smoke_not_webcam_trial", "fixtures": results,
                  "video_rows": len(steps), "video_active_rows": active,
                  "note": "Unknown K and palm width: valid PnP is not evidence of accurate metric depth."}
        dump(args.output, result)
        print(result)


if __name__ == "__main__":
    main()
