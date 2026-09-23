from types import SimpleNamespace
import numpy as np
import pytest
pytest.importorskip("pybullet")
from gesture.cli import parser
from gesture.control import Settings, Command
from gesture.simulation import Simulation
from gesture.runtime import run, replay, reprocess, evaluate
from gesture.data import verify_episode, export_npz

pytestmark = pytest.mark.integration


def test_simulation_initial_state_render_and_rejection():
    with Simulation(Settings()) as sim:
        state = sim.state()
        assert np.isfinite(state.joints).all() and state.joints.shape == (9,)
        assert np.linalg.norm(state.position - [.5, 0., .4]) < .02
        assert sim.render(320, 240).shape == (240, 320, 3)
        sim.step(Command([2., 0., .4], [1., 0., 0., 0.], .08, True))
        assert sim.rejections == 1 and sim.reason == "outside_workspace"


@pytest.mark.parametrize("task,seconds", [("reach", 9), ("path", 23), ("pick-place", 30)])
def test_end_to_end_task_and_command_replay(tmp_path, task, seconds):
    out = tmp_path / task
    args = parser().parse_args(["demo", "--task", task, "--seconds", str(seconds),
                               "--headless", "--output", str(out), "--assert-success"])
    summary = run(args)
    assert summary["success"]
    assert verify_episode(out)
    result = replay(SimpleNamespace(episode=out, mode="command", headless=True, tolerance=1e-5))
    assert result["max_ee_replay_error_m"] < 1e-5
    export_npz(out, tmp_path / f"{task}.npz")
    if task == "reach":
        reprocess(SimpleNamespace(episode=out, output=tmp_path / "unfiltered", filter="none"))
        assert verify_episode(tmp_path / "unfiltered")
        metrics = evaluate(out, tmp_path / "evaluation")
        assert metrics["synthetic_pose_rmse_m"] < 1e-4
        assert metrics["pose_valid_fraction"] > .99
