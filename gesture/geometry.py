"""Metric geometry with explicit local-hand, optical-camera, and robot frames."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Optical: +x right, +y down, +z forward. Robot: +x forward, +y left, +z up.
CAMERA_TO_ROBOT = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
PALM = np.array([0, 1, 2, 5, 9, 13, 17])
DOWN_QUAT = np.array([1., 0., 0., 0.])  # xyzw everywhere, radians, metres.


def array(value, shape):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Expected finite array with shape {shape}; got {value.shape}")
    return value


def unit(value):
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("degenerate_palm")
    return value / norm


def bounded(value, length):
    return value * min(1., max(0., length) / max(float(np.linalg.norm(value)), 1e-12))


def palm_frame(points):
    x = unit(points[5] - points[17])
    y = points[9] - points[0]
    y = unit(y - x * np.dot(x, y))
    return np.column_stack([x, y, unit(np.cross(x, y))])


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple = (0., 0., 0., 0., 0.)
    calibrated: bool = False

    def __post_init__(self):
        if self.width < 16 or self.height < 16 or not np.isfinite(
                [self.fx, self.fy, self.cx, self.cy, *self.distortion]).all():
            raise ValueError("Invalid intrinsics")
        if min(self.fx, self.fy) <= 0 or len(self.distortion) not in (4, 5, 8, 12, 14):
            raise ValueError("Invalid focal length or distortion coefficients")

    @property
    def matrix(self):
        return np.array([[self.fx, 0., self.cx], [0., self.fy, self.cy], [0., 0., 1.]])

    @classmethod
    def approximate(cls, width=640, height=480, hfov=65.):
        if not 10 < hfov < 160:
            raise ValueError("Horizontal FOV must be 10..160 degrees")
        focal = width / (2. * np.tan(np.deg2rad(hfov) / 2.))
        return cls(width, height, focal, focal, width / 2., height / 2.)

    def resized(self, width, height):
        if abs(width / height - self.width / self.height) > .015:
            raise ValueError("Camera aspect ratio changed; recalibrate (cropping is not scaling K)")
        sx, sy = width / self.width, height / self.height
        return Intrinsics(width, height, self.fx * sx, self.fy * sy, self.cx * sx,
                          self.cy * sy, self.distortion, self.calibrated)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class Observation:
    timestamp: float
    uv: np.ndarray            # 21x2 original unmirrored pixels.
    local_xyz: np.ndarray     # 21x3 model-local metres, NOT global translation.
    handedness: str = "unknown"
    inference_ms: float = 0.
    frame_id: int = 0

    def __post_init__(self):
        self.uv = array(self.uv, (21, 2))
        self.local_xyz = array(self.local_xyz, (21, 3))
        if not np.isfinite([self.timestamp, self.inference_ms]).all():
            raise ValueError("Invalid observation timestamp / inference time")


@dataclass
class HandPose:
    timestamp: float
    position: np.ndarray
    rotation: np.ndarray
    pinch: float
    reprojection_px: float
    handedness: str
    frame_id: int


class PoseEstimator:
    """Seven-palm-point SQPnP with measured scale prior and rejection gates.

    Reprojection residual is NOT 3-D accuracy. Monocular metric depth is conditioned
    on the supplied intrinsics and palm width; neural shape errors remain.
    """
    def __init__(self, intrinsics, palm_width=.08, max_error_px=10.):
        if not .035 <= palm_width <= .14 or max_error_px <= 0:
            raise ValueError("Palm width must be .035.. .14 m and threshold > 0")
        self.k, self.palm_width, self.max_error_px = intrinsics, palm_width, max_error_px
        self.reason = "not_started"

    def estimate(self, obs):
        self.reason = "no_hand"
        if obs is None:
            return None
        try:
            xyz, uv = array(obs.local_xyz, (21, 3)), array(obs.uv, (21, 2))
            width = np.linalg.norm(xyz[5] - xyz[17])
            if not .015 < width < .2:
                raise ValueError("implausible_model_scale")
            xyz = (xyz - xyz[0]) * self.palm_width / width
            image = np.ascontiguousarray(uv[PALM])
            if (np.min(image) < -5 or np.any(image[:, 0] > self.k.width + 5)
                    or np.any(image[:, 1] > self.k.height + 5)):
                raise ValueError("palm_out_of_frame")
            area = cv2.contourArea(cv2.convexHull(image.astype(np.float32)))
            if area < 160 or np.linalg.norm(uv[5] - uv[17]) < 16:
                raise ValueError("palm_too_small_or_edge_on")
            model = np.ascontiguousarray(xyz[PALM])
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                model, image, self.k.matrix, np.asarray(self.k.distortion), flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                raise ValueError("pnp_failed")
            candidates = []
            for rv, tv in zip(rvecs, tvecs):
                cam = xyz @ cv2.Rodrigues(rv)[0].T + tv.reshape(3)
                if np.min(cam[:, 2]) <= .08 or not .12 < cam[0, 2] < 2.5:
                    continue
                projected = cv2.projectPoints(model, rv, tv, self.k.matrix,
                                              np.asarray(self.k.distortion))[0].reshape(-1, 2)
                error = float(np.sqrt(np.mean(np.sum((projected - image) ** 2, axis=1))))
                candidates.append((error, cam))
            if not candidates:
                raise ValueError("invalid_depth")
            error, cam = min(candidates, key=lambda item: item[0])
            if not np.isfinite(error) or error > self.max_error_px:
                raise ValueError("high_reprojection_error")
            frame = palm_frame(cam)
            self.reason = "valid"
            return HandPose(obs.timestamp, cam[0].copy(), frame,
                            float(np.linalg.norm(xyz[4] - xyz[8]) / self.palm_width),
                            error, obs.handedness, obs.frame_id)
        except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
            self.reason = str(exc)
            return None


def rotate_towards(current, target, max_angle):
    delta = Rotation.from_matrix(target @ current.T).as_rotvec()
    return Rotation.from_rotvec(bounded(delta, max_angle)).as_matrix() @ current
