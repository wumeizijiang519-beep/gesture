"""Local MediaPipe Tasks and bounded latest-frame capture; no cloud inference."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
import platform
import threading
import time
import urllib.request
import cv2
import numpy as np
from .geometry import Observation

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def verify_model(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing {path}. Run: gesture download-model --output {path}")
    digest = sha256(path)
    if digest != MODEL_SHA256:
        raise ValueError(f"Model checksum mismatch. Expected {MODEL_SHA256}, got {digest}")
    return digest


def download_model(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return verify_model(path)
    tmp = path.with_suffix(".download")
    if tmp.exists():
        raise FileExistsError(f"Partial download exists: {tmp}; inspect/remove it before retrying")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, tmp.open("xb") as out:
            total = 0
            while data := response.read(1024 * 1024):
                total += len(data)
                if total > 20 * 1024 * 1024:
                    raise ValueError("Unexpected model size")
                out.write(data)
        digest = verify_model(tmp)
        tmp.replace(path)
        return digest
    finally:
        tmp.unlink(missing_ok=True)


class Detector:
    def __init__(self, model_path):
        self.model_sha256 = verify_model(model_path)
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path),
                                            delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=.6, min_hand_presence_confidence=.6,
            min_tracking_confidence=.6)
        self.model = vision.HandLandmarker.create_from_options(options)
        self.last_ms, self.reason = -1, "not_started"

    def detect(self, bgr, timestamp, frame_id):
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError("Detector requires uint8 BGR")
        start = time.perf_counter()
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.last_ms = max(self.last_ms + 1, int(round(timestamp * 1000)))
        result = self.model.detect_for_video(image, self.last_ms)
        # Never silently switch between two operators.
        if len(result.hand_landmarks) != 1:
            self.reason = "no_hand" if not result.hand_landmarks else "multiple_hands"
            return None
        height, width = bgr.shape[:2]
        uv = np.array([[point.x * width, point.y * height] for point in result.hand_landmarks[0]])
        xyz = np.array([[point.x, point.y, point.z] for point in result.hand_world_landmarks[0]])
        hand = result.handedness[0][0].category_name
        self.reason = "valid"
        # The handedness score is deliberately NOT used as landmark confidence.
        return Observation(timestamp, uv, xyz, hand, (time.perf_counter() - start) * 1000, frame_id)

    def close(self):
        self.model.close()


def open_camera(index, width=640, height=480, fps=30):
    if index < 0:
        raise ValueError("Camera index must be nonnegative")
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera {index}; check privacy permissions / competing applications")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Some backends ignore this; latest-value slots also bound buffering.
    return cap


@dataclass
class Packet:
    frame: np.ndarray
    timestamp: float
    frame_id: int
    observation: Observation | None
    reason: str


class Webcam:
    """Separate capture and inference threads; two latest-value slots, never an unbounded queue.

    Host read-completion time is not the sensor exposure timestamp. A blocked device
    cannot block the controller's stale-data watchdog. Camera threads own their resources.
    """
    def __init__(self, index, model, origin, width=640, height=480, fps=30):
        self.cap = open_camera(index, width, height, fps)
        self.origin, self.model_path = origin, model
        self.latest_frame = self.latest_packet = None
        self.error = None
        self.stop_event, self.lock = threading.Event(), threading.Lock()
        self.capture_thread = threading.Thread(target=self._capture, name="capture", daemon=True)
        self.inference_thread = threading.Thread(target=self._infer, name="inference", daemon=True)
        self.capture_thread.start()
        self.inference_thread.start()

    def _capture(self):
        try:
            frame_id = 0
            while not self.stop_event.is_set():
                ok, image = self.cap.read()
                timestamp = time.monotonic() - self.origin
                if not ok:
                    raise RuntimeError("Camera read failed / disconnected")
                with self.lock:
                    self.latest_frame = (image, timestamp, frame_id)
                frame_id += 1
        except Exception as exc:
            self.error = f"Capture: {exc}"
        finally:
            self.cap.release()

    def _infer(self):
        detector = None
        try:
            detector = Detector(self.model_path)
            last = -1
            while not self.stop_event.is_set():
                with self.lock:
                    sample = self.latest_frame
                if sample is None or sample[2] == last:
                    self.stop_event.wait(.002)
                    continue
                image, timestamp, last = sample
                obs = detector.detect(image, timestamp, last)
                with self.lock:
                    self.latest_packet = Packet(image, timestamp, last, obs, detector.reason)
        except Exception as exc:
            self.error = f"Inference: {exc}"
        finally:
            if detector is not None:
                detector.close()

    def poll(self):
        with self.lock:
            return self.latest_packet

    def close(self):
        self.stop_event.set()
        self.capture_thread.join(timeout=2.)
        self.inference_thread.join(timeout=2.)


class Video:
    """Deterministic offline video sampling at the control clock, with no frame seeking."""
    def __init__(self, path, model):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot decode video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(self.fps) or not 1 <= self.fps <= 240:
            self.cap.release()
            raise ValueError("Video must provide a valid constant frame rate (1..240 Hz)")
        try:
            self.detector = Detector(model)
        except Exception:
            self.cap.release()
            raise
        self.next_id, self.packet, self.eof = 0, None, False

    def poll(self, timestamp):
        # CFR assumption is recorded; VFR video must first be converted explicitly.
        wanted = int(np.floor(timestamp * self.fps + 1e-7))
        last = None
        while self.next_id <= wanted:
            ok, image = self.cap.read()
            if not ok:
                self.eof = True
                break
            last = (image, self.next_id / self.fps, self.next_id)
            self.next_id += 1
        if last is not None:
            image, ts, frame_id = last
            obs = self.detector.detect(image, ts, frame_id)
            self.packet = Packet(image, ts, frame_id, obs, self.detector.reason)
        return self.packet

    def close(self):
        self.cap.release()
        self.detector.close()


def draw_hand(frame, obs, mirror=False):
    image = frame.copy()
    if obs is not None:
        for a, b in CONNECTIONS:
            pa, pb = tuple(np.round(obs.uv[a]).astype(int)), tuple(np.round(obs.uv[b]).astype(int))
            cv2.line(image, pa, pb, (110, 230, 130), 2, cv2.LINE_AA)
        for uv in obs.uv:
            cv2.circle(image, tuple(np.round(uv).astype(int)), 3, (0, 210, 255), -1, cv2.LINE_AA)
    return cv2.flip(image, 1) if mirror else image
