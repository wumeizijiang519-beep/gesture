"""Relative retargeting, clutch, signal filtering, and latched simulated stops."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import CAMERA_TO_ROBOT, DOWN_QUAT, array, bounded, rotate_towards


@dataclass
class Settings:
    seed: int = 7
    hz: int = 30
    task: str = "reach"
    orientation: bool = False
    filter: str = "one-euro"
    gain: float = 1.
    min_cutoff: float = 1.5
    beta: float = 5.
    max_speed: float = .25
    max_acceleration: float = 1.
    max_angular_speed: float = 1.
    max_joint_speed: float = 1.5
    stale_seconds: float = .25
    max_hand_speed: float = 3.
    workspace_min: list = field(default_factory=lambda: [.25, -.35, .035])
    workspace_max: list = field(default_factory=lambda: [.75, .35, .65])

    def __post_init__(self):
        if self.hz not in (20, 24, 30, 40, 60):
            raise ValueError("Control Hz must divide the 240 Hz physics clock")
        if self.task not in ("reach", "path", "pick-place") or self.filter not in ("one-euro", "none"):
            raise ValueError("Unknown task / filter")
        for name in ("gain", "min_cutoff", "max_speed", "max_acceleration", "max_angular_speed",
                     "max_joint_speed", "stale_seconds", "max_hand_speed"):
            if not 0 < getattr(self, name) < 20:
                raise ValueError(f"Invalid setting: {name}")
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be nonnegative")
        if np.any(array(self.workspace_min, (3,)) >= array(self.workspace_max, (3,))):
            raise ValueError("Empty workspace")


class OneEuro:
    def __init__(self, min_cutoff=1.5, beta=5., derivative_cutoff=1.):
        self.min_cutoff, self.beta, self.derivative_cutoff = min_cutoff, beta, derivative_cutoff
        self.x = self.raw = self.derivative = self.time = None

    @staticmethod
    def alpha(cutoff, dt):
        return 1. / (1. + 1. / (2. * np.pi * cutoff * dt))

    def __call__(self, value, timestamp):
        value = np.asarray(value, dtype=float)
        if not np.isfinite(value).all() or not np.isfinite(timestamp):
            raise ValueError("Non-finite filter input")
        if self.time is None:
            self.x, self.raw, self.derivative = value.copy(), value.copy(), np.zeros_like(value)
        elif timestamp > self.time:
            dt = timestamp - self.time
            a = self.alpha(self.derivative_cutoff, dt)
            self.derivative = a * (value - self.raw) / dt + (1. - a) * self.derivative
            a = self.alpha(self.min_cutoff + self.beta * np.abs(self.derivative), dt)
            self.x = a * value + (1. - a) * self.x
            self.raw = value.copy()
        else:
            return self.x.copy()
        self.time = timestamp
        return self.x.copy()


class State(str, Enum):
    DISARMED = "DISARMED"
    ACTIVE = "ACTIVE"
    LOST = "LOST"
    ESTOP = "ESTOP"


@dataclass
class Command:
    position: np.ndarray
    quaternion: np.ndarray
    grip: float
    active: bool

    def __post_init__(self):
        self.position = array(self.position, (3,))
        self.quaternion = array(self.quaternion, (4,))
        if abs(np.linalg.norm(self.quaternion) - 1.) > .01 or not 0 <= self.grip <= .080001:
            raise ValueError("Invalid quaternion / gripper width")


class Controller:
    def __init__(self, cfg):
        self.cfg, self.state, self.calibrated = cfg, State.DISARMED, False
        self.message = "Show ONE hand; C then SPACE"
        self.last_pose = None
        self.hand_anchor = self.robot_anchor = self.rotation_anchor = self.robot_rot_anchor = None
        self.target = np.array([.5, 0., .4])
        self.rotation = Rotation.from_quat(DOWN_QUAT).as_matrix()
        self.grip, self.velocity = .08, np.zeros(3)
        self.position_filter = OneEuro(cfg.min_cutoff, cfg.beta)
        self.rejected = self.clips = 0

    def stop(self, reason, emergency=False):
        if self.state == State.ESTOP and not emergency:
            return
        self.state = State.ESTOP if emergency else State.LOST
        self.message, self.velocity = reason, np.zeros(3)

    def observe(self, pose):
        if pose is None:
            self.last_pose = None
            if self.state == State.ACTIVE:
                self.stop("Tracking lost; SPACE required to re-anchor")
            return
        if self.last_pose is not None:
            dt = pose.timestamp - self.last_pose.timestamp
            if dt <= 0:
                return
            speed = np.linalg.norm(pose.position - self.last_pose.position) / dt
            turn = Rotation.from_matrix(pose.rotation @ self.last_pose.rotation.T).magnitude() / dt
            if self.state == State.ACTIVE and (pose.handedness != self.last_pose.handedness
                                               or speed > self.cfg.max_hand_speed or turn > 15.):
                self.rejected += 1
                self.stop("Hand identity / pose jump rejected; SPACE to re-anchor")
        self.last_pose = pose

    def event(self, event, ee_position, ee_quaternion, now):
        if event == "estop":
            self.stop("STOP latched. R clears latch, C then SPACE resumes.", emergency=True)
            return
        if event == "reset":
            self.state, self.calibrated = State.DISARMED, False
            self.message, self.velocity = "Reset: C then SPACE", np.zeros(3)
            return
        if self.state == State.ESTOP:
            return
        if event == "toggle" and self.state == State.ACTIVE:
            self.state, self.message, self.velocity = State.DISARMED, "Clutch released", np.zeros(3)
            return
        if self.last_pose is None or not 0 <= now - self.last_pose.timestamp <= self.cfg.stale_seconds:
            self.message = "No fresh valid hand pose"
            return
        if event == "calibrate":
            self.calibrated, self.state = True, State.DISARMED
            self.message = "Neutral accepted. SPACE to engage"
        elif event == "toggle" and self.calibrated:
            self.hand_anchor = self.last_pose.position.copy()
            self.rotation_anchor = self.last_pose.rotation.copy()
            self.robot_anchor = array(ee_position, (3,)).copy()
            self.robot_rot_anchor = Rotation.from_quat(ee_quaternion).as_matrix()
            self.target, self.rotation = self.robot_anchor.copy(), self.robot_rot_anchor.copy()
            self.velocity[:] = 0.
            self.position_filter = OneEuro(self.cfg.min_cutoff, self.cfg.beta)
            self.state, self.message = State.ACTIVE, "Hand control active"
        elif event == "toggle":
            self.message = "Press C first"

    def step(self, now, dt):
        if not np.isfinite([now, dt]).all() or not 0 < dt <= .1:
            self.stop("Invalid control timestep", emergency=True)
        if self.state == State.ACTIVE:
            pose = self.last_pose
            if pose is None or not 0 <= now - pose.timestamp <= self.cfg.stale_seconds:
                self.stop("Camera timeout; SPACE required to resume")
            else:
                pos = pose.position
                if self.cfg.filter == "one-euro":
                    pos = self.position_filter(pos, pose.timestamp)
                desired = self.robot_anchor + self.cfg.gain * CAMERA_TO_ROBOT @ (pos - self.hand_anchor)
                goal = np.clip(desired, self.cfg.workspace_min, self.cfg.workspace_max)
                self.clips += int(np.linalg.norm(goal - desired) > 1e-8)
                # Braking-speed envelope; emergency/workspace bounds take priority over smoothness.
                delta = goal - self.target
                speed = min(self.cfg.max_speed, np.sqrt(2 * self.cfg.max_acceleration * np.linalg.norm(delta)))
                wanted_v = bounded(delta / dt, speed)
                self.velocity += bounded(wanted_v - self.velocity, self.cfg.max_acceleration * dt)
                advance = self.velocity * dt
                if np.dot(advance, delta) > 0 and np.linalg.norm(advance) > np.linalg.norm(delta):
                    advance, self.velocity = delta, np.zeros(3)
                self.target = np.clip(self.target + advance, self.cfg.workspace_min, self.cfg.workspace_max)
                if self.cfg.orientation:
                    rel = CAMERA_TO_ROBOT @ pose.rotation @ self.rotation_anchor.T @ CAMERA_TO_ROBOT.T
                    rv = bounded(Rotation.from_matrix(rel).as_rotvec(), np.deg2rad(60.))
                    goal_rot = Rotation.from_rotvec(rv).as_matrix() @ self.robot_rot_anchor
                    self.rotation = rotate_towards(self.rotation, goal_rot, self.cfg.max_angular_speed * dt)
                opening = .08 * np.clip((pose.pinch - .20) / .70, 0., 1.)
                self.grip = float(np.clip(self.grip + np.clip(opening - self.grip, -.12 * dt, .12 * dt), 0., .08))
        return Command(self.target.copy(), Rotation.from_matrix(self.rotation).as_quat(),
                       self.grip, self.state == State.ACTIVE)
