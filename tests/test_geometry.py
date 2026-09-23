import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from gesture.geometry import (Intrinsics, PoseEstimator, Observation, CAMERA_TO_ROBOT,
                              palm_frame, rotate_towards, array)
from gesture.synthetic import observation, hand_template


@pytest.mark.parametrize("depth", [.35, .55, .85, 1.2])
def test_metric_pnp_translation(depth):
    k = Intrinsics.approximate()
    expected = np.array([.02, .01, depth])
    pose = PoseEstimator(k).estimate(observation(k, expected, 0.))
    assert pose is not None
    np.testing.assert_allclose(pose.position, expected, atol=1e-5)
    assert pose.reprojection_px < 1e-4


@pytest.mark.parametrize("angle", [-25, 0, 25])
def test_rotated_palm(angle):
    k = Intrinsics.approximate()
    rotation = Rotation.from_euler("y", angle, degrees=True).as_matrix()
    obs = observation(k, [0., .02, .7], 0., rotation=rotation)
    pose = PoseEstimator(k).estimate(obs)
    assert pose is not None
    expected = rotation @ palm_frame(hand_template())
    np.testing.assert_allclose(pose.rotation, expected, atol=1e-5)
    assert np.linalg.det(pose.rotation) == pytest.approx(1.)


def test_local_origin_not_global_translation():
    k = Intrinsics.approximate()
    a = observation(k, [0., 0., .65], 0.)
    b = observation(k, [.05, .02, .95], 1.)
    np.testing.assert_array_equal(a.local_xyz, b.local_xyz)
    pa, pb = PoseEstimator(k).estimate(a), PoseEstimator(k).estimate(b)
    np.testing.assert_allclose(pb.position - pa.position, [.05, .02, .30], atol=1e-5)


def test_scale_prior_is_explicit():
    k = Intrinsics.approximate()
    obs = observation(k, [0., 0., .7], 0.)
    p = PoseEstimator(k, palm_width=.10).estimate(obs)
    np.testing.assert_allclose(p.position, [0., 0., .875], atol=1e-5)


def test_invalid_or_out_of_frame():
    est = PoseEstimator(Intrinsics.approximate())
    assert est.estimate(None) is None
    assert est.estimate(observation(est.k, [2., 0., .7], 0.)) is None
    assert est.estimate(Observation(0., np.ones((21, 2)), hand_template())) is None


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_rejected(bad):
    with pytest.raises(ValueError):
        Observation(0., np.full((21, 2), bad), hand_template())
    with pytest.raises(ValueError):
        Intrinsics(640, 480, bad, 500., 320., 240.)


def test_intrinsics_scaling_roundtrip(tmp_path):
    k = Intrinsics.approximate()
    k.save(tmp_path / "k.json")
    np.testing.assert_allclose(Intrinsics.load(tmp_path / "k.json").matrix, k.matrix)
    bigger = k.resized(1280, 960)
    np.testing.assert_allclose(bigger.matrix[:2], k.matrix[:2] * 2)
    with pytest.raises(ValueError):
        k.resized(1280, 720)


def test_frame_rotation_and_angular_cap():
    np.testing.assert_allclose(CAMERA_TO_ROBOT.T @ CAMERA_TO_ROBOT, np.eye(3))
    assert np.linalg.det(CAMERA_TO_ROBOT) == pytest.approx(1.)
    result = rotate_towards(np.eye(3), Rotation.from_euler("x", 1.).as_matrix(), .02)
    assert Rotation.from_matrix(result).magnitude() == pytest.approx(.02)


def test_bad_shape_and_width():
    with pytest.raises(ValueError):
        array([1., 2.], (3,))
    with pytest.raises(ValueError):
        PoseEstimator(Intrinsics.approximate(), palm_width=8.)
