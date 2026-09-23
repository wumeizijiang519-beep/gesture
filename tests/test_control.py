import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from gesture.geometry import HandPose, DOWN_QUAT
from gesture.control import Controller, Settings, State, OneEuro, Command


def pose(timestamp=0., position=None, hand="right"):
    return HandPose(timestamp, np.array([0., 0., .7]) if position is None else np.array(position),
                    np.eye(3), .9, 0., hand, int(timestamp * 1000))


def armed(cfg=None):
    ctrl = Controller(cfg or Settings())
    ctrl.observe(pose())
    ctrl.event("calibrate", np.array([.5, 0., .4]), DOWN_QUAT, 0.)
    ctrl.event("toggle", np.array([.5, 0., .4]), DOWN_QUAT, 0.)
    assert ctrl.state == State.ACTIVE
    return ctrl


def test_requires_explicit_neutral_and_clutch():
    ctrl = Controller(Settings())
    ctrl.observe(pose())
    ctrl.event("toggle", [.5, 0., .4], DOWN_QUAT, 0.)
    assert ctrl.state == State.DISARMED
    assert not ctrl.step(0., 1 / 30).active


def test_no_jump_on_reengagement():
    ctrl = armed()
    ctrl.event("toggle", [.5, 0., .4], DOWN_QUAT, 0.)
    ctrl.observe(pose(1., [.2, -.1, .9]))
    ctrl.event("toggle", [.5, 0., .4], DOWN_QUAT, 1.)
    cmd = ctrl.step(1., 1 / 30)
    np.testing.assert_allclose(cmd.position, [.5, 0., .4])


@pytest.mark.parametrize("failure", ["lost", "stale", "identity", "jump"])
def test_loss_requires_manual_resume(failure):
    ctrl = armed()
    if failure == "lost":
        ctrl.observe(None)
    elif failure == "identity":
        ctrl.observe(pose(.03, hand="left"))
    elif failure == "jump":
        ctrl.observe(pose(.03, [1., 0., .7]))
    else:
        ctrl.step(.30, 1 / 30)
    assert ctrl.state == State.LOST
    ctrl.observe(pose(.4))
    assert not ctrl.step(.4, 1 / 30).active


def test_estop_is_latched():
    ctrl = armed()
    ctrl.event("estop", [.5, 0., .4], DOWN_QUAT, 0.)
    ctrl.observe(None)
    ctrl.observe(pose(.1))
    for event in ("calibrate", "toggle"):
        ctrl.event(event, [.5, 0., .4], DOWN_QUAT, .1)
    assert ctrl.state == State.ESTOP
    ctrl.event("reset", [.5, 0., .4], DOWN_QUAT, .1)
    assert ctrl.state == State.DISARMED and not ctrl.calibrated


def test_speed_and_workspace_bounds():
    ctrl = armed()
    previous = ctrl.target.copy()
    for i in range(1, 180):
        ctrl.observe(pose(i / 30, [0., 0., .7 + .004 * i]))
        cmd = ctrl.step(i / 30, 1 / 30)
        assert np.linalg.norm(cmd.position - previous) <= ctrl.cfg.max_speed / 30 + 1e-10
        assert np.all(cmd.position >= ctrl.cfg.workspace_min)
        assert np.all(cmd.position <= ctrl.cfg.workspace_max)
        previous = cmd.position.copy()
    assert ctrl.clips > 0


def test_orientation_speed_limited():
    ctrl = armed(Settings(orientation=True))
    previous = Rotation.from_quat(DOWN_QUAT)
    for i in range(1, 60):
        p = pose(i / 30)
        p.rotation = Rotation.from_euler("y", .02 * i).as_matrix()
        ctrl.observe(p)
        cmd = ctrl.step(i / 30, 1 / 30)
        rot = Rotation.from_quat(cmd.quaternion)
        assert (rot * previous.inv()).magnitude() <= 1 / 30 + 1e-10
        previous = rot


def test_filter_reduces_stationary_jitter():
    rng = np.random.default_rng(7)
    raw = rng.normal(0., .003, (300, 3))
    filt = OneEuro()
    output = np.array([filt(x, i / 30) for i, x in enumerate(raw)])
    assert np.std(output[60:]) < np.std(raw[60:]) * .65
    np.testing.assert_array_equal(filt([100, 100, 100], 0.), output[-1])


def test_bad_timing_latches_stop_and_invalid_command_rejected():
    ctrl = armed()
    ctrl.step(0., -1.)
    assert ctrl.state == State.ESTOP
    with pytest.raises(ValueError):
        Command([0., 0., 0.], [0., 0., 0., 0.], .08, True)
    with pytest.raises(ValueError):
        Settings(hz=29)
    with pytest.raises(ValueError):
        Settings(gain=float("nan"))
