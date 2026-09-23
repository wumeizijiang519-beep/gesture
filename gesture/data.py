"""Versioned JSONL episodes, checksums, provenance, bounded optional video, safe NPZ."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import cv2
import numpy as np
from .vision import sha256

SCHEMA = "gesture.episode.v1"


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def dump(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(content, default=serial, allow_nan=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def provenance():
    versions = {}
    for name in ("gesture-rgb-teleop", "numpy", "scipy", "opencv-contrib-python", "pybullet", "mediapipe"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                           text=True, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unknown (archive / non-git installation)"
    return {"python": platform.python_version(), "platform": platform.platform(),
            "versions": versions, "git_revision": revision}


class Recorder:
    def __init__(self, folder, metadata, save_video=False, fps=30, max_mb=512):
        self.path = Path(folder)
        if self.path.exists():
            raise FileExistsError(f"Episode already exists; choose a new output directory: {self.path}")
        if max_mb < 1 or not np.isfinite(max_mb):
            raise ValueError("max-mb must be finite and >= 1")
        self.path.mkdir(parents=True)
        dump(self.path / "meta.json", {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                                      "provenance": provenance(), **metadata})
        self.file = (self.path / "steps.jsonl").open("x", encoding="utf-8")
        self.save_video, self.fps, self.limit = save_video, fps, int(max_mb * 1024 * 1024)
        self.writer = None
        self.video_index, self.video_shape, self.video_frame_id = -1, None, None
        self.rows, self.closed = 0, False

    def append(self, row, frame=None):
        if self.closed:
            raise RuntimeError("Recorder is closed")
        row = dict(row)
        frame_id = row.get("frame_id")
        if self.save_video and frame is not None and frame_id != self.video_frame_id:
            if self.writer is None:
                h, w = frame.shape[:2]
                self.video_shape = (h, w)
                self.writer = cv2.VideoWriter(str(self.path / "rgb.avi"),
                                               cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (w, h))
                if not self.writer.isOpened():
                    raise RuntimeError("MJPG writer unavailable; run without --save-video")
            if frame.shape[:2] != self.video_shape:
                raise ValueError("Video resolution changed within episode")
            self.writer.write(frame)
            self.video_index += 1
            self.video_frame_id = frame_id
        row["video_frame_index"] = self.video_index if self.save_video and self.video_index >= 0 else None
        self.file.write(json.dumps(row, default=serial, allow_nan=False, separators=(",", ":")) + "\n")
        self.rows += 1
        if self.rows % 15 == 0:
            self.file.flush()
            if sum(p.stat().st_size for p in self.path.iterdir() if p.is_file()) > self.limit:
                raise RuntimeError("Episode disk budget reached; stopping, not deleting recorded data")

    def close(self, summary=None):
        if self.closed:
            return
        self.closed = True
        self.file.flush()
        self.file.close()
        if self.writer is not None:
            self.writer.release()
        dump(self.path / "summary.json", {"rows": self.rows, **(summary or {})})
        dump(self.path / "manifest.json", {p.name: sha256(p) for p in sorted(self.path.iterdir())
                                           if p.is_file() and p.name != "manifest.json"})


def read_episode(folder):
    path = Path(folder)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA:
        raise ValueError("Unsupported episode schema")
    rows = []
    with (path / "steps.jsonl").open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupt JSONL at line {index}; no silent row deletion") from exc
    if not rows:
        raise ValueError("Empty episode")
    times = np.array([row["sim_time"] for row in rows])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Episode timestamps must strictly increase")
    return meta, rows


def verify_episode(folder):
    path = Path(folder)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if not {"meta.json", "steps.jsonl", "summary.json"}.issubset(manifest):
        raise ValueError("Incomplete episode manifest")
    for name, expected in manifest.items():
        if Path(name).name != name or sha256(path / name) != expected:
            raise ValueError(f"Episode integrity failure: {name}")
    return True


def export_npz(folder, output):
    verify_episode(folder)
    _, rows = read_episode(folder)
    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("Export path must end in .npz")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output,
        sim_time=np.array([r["sim_time"] for r in rows]),
        capture_time=np.array([r["capture_time"] if r["capture_time"] is not None else np.nan for r in rows]),
        uv=np.array([r["observation"]["uv"] if r["observation"] else np.full((21, 2), np.nan) for r in rows]),
        hand_local_xyz=np.array([r["observation"]["local_xyz"] if r["observation"] else
                                 np.full((21, 3), np.nan) for r in rows]),
        ee_position=np.array([r["robot"]["position"] for r in rows]),
        ee_quaternion_xyzw=np.array([r["robot"]["quaternion"] for r in rows]),
        qpos=np.array([r["robot"]["joints"] for r in rows]),
        action_ee_position=np.array([r["command"]["position"] for r in rows]),
        action_quaternion_xyzw=np.array([r["command"]["quaternion"] for r in rows]),
        action_grip=np.array([r["command"]["grip"] for r in rows]),
        applied_motor_targets=np.array([r["motor_targets"] for r in rows]),
        active=np.array([r["command"]["active"] for r in rows], dtype=bool),
        state=np.array([r["state"] for r in rows], dtype="U16"))
