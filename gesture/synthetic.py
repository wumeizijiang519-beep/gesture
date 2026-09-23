"""Seeded geometric fixtures, explicitly not real-camera observations or user trials."""
from __future__ import annotations
import cv2
import numpy as np
from .geometry import CAMERA_TO_ROBOT, Observation
from .simulation import HOME


def hand_template(pinch=.9, palm_width=.08):
    points = np.zeros((21, 3), dtype=float)
    points[0] = [0., .025, 0.]
    points[1:5] = [[-.026, .016, -.012], [-.043, -.001, -.013],
                   [-.061, -.010, -.008], [-.073, -.028, -.003]]
    for start, x, y, z in [(5, -.040, -.027, .004), (9, -.013, -.043, 0.),
                           (13, .014, -.037, .002), (17, .040, -.021, .009)]:
        for j in range(4):
            points[start + j] = [x + j * .001, y - j * .022, z - j * .001]
    points -= points[0]
    points *= palm_width / np.linalg.norm(points[5] - points[17])
    # Fingertips change pinch, while palm PnP geometry stays fixed.
    points[4] = points[8] + np.array([-pinch * palm_width, 0., 0.])
    return points


def observation(k, position, timestamp, pinch=.9, noise_px=0., rng=None, rotation=None, frame_id=0):
    xyz = hand_template(pinch)
    rotation = np.eye(3) if rotation is None else rotation
    rv = cv2.Rodrigues(rotation)[0]
    uv = cv2.projectPoints(xyz, rv, np.asarray(position, dtype=float), k.matrix,
                          np.asarray(k.distortion))[0].reshape(21, 2)
    if noise_px:
        if rng is None:
            raise ValueError("A seeded RNG is required when adding noise")
        uv += rng.normal(0., noise_px, uv.shape)
    return Observation(timestamp, uv, xyz, "synthetic", 0., frame_id)


def demo_target(task, t, goal):
    if task == "reach":
        f = np.clip(t / 4., 0., 1.)
        return HOME * (1. - f) + goal * f, .9
    if task == "path":
        if t < 3.:
            f = t / 3.
            return HOME * (1. - f) + np.array([.5, .1, .3]) * f, .9
        angle = min(t - 3., 16.) * 2. * np.pi / 16.
        return np.array([.5, .1 * np.cos(angle), .3 + .08 * np.sin(angle)]), .9
    points = [(0., HOME, .9), (3., [.5, -.13, .24], .9),
              (6., [.5, -.13, .039], .9), (7., [.5, -.13, .039], .9),
              (8.5, [.5, -.13, .039], .20), (10., [.5, -.13, .039], .20),
              (13., [.5, -.13, .26], .20), (17., [.5, .18, .26], .20),
              (20., [.5, .18, .04], .20), (21., [.5, .18, .04], .20),
              (22.5, [.5, .18, .04], .9), (24., [.5, .18, .04], .9),
              (27., [.5, .18, .25], .9), (30., [.5, .18, .25], .9)]
    for (a, pa, ga), (b, pb, gb) in zip(points, points[1:]):
        if t <= b:
            f = np.clip((t - a) / (b - a), 0., 1.)
            return np.asarray(pa) * (1 - f) + np.asarray(pb) * f, ga * (1 - f) + gb * f
    return np.asarray(points[-1][1]), .9


def camera_target(robot_target, gain=1.):
    return np.array([0., 0., .85]) + CAMERA_TO_ROBOT.T @ (robot_target - HOME) / gain
