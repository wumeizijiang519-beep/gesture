"""End-to-end application, synthetic integration run, recorded-command replay, and ablation."""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .control import Settings, Controller, Command
from .data import Recorder, read_episode, verify_episode, dump
from .geometry import Intrinsics, PoseEstimator, Observation
from .simulation import Simulation
from .synthetic import observation, camera_target, demo_target
from .vision import Webcam, Video, Packet, draw_hand, verify_model, sha256

KEYS = {ord('c'): "calibrate", 32: "toggle", ord('x'): "estop", ord('r'): "reset"}


def config_from_args(args):
    return Settings(seed=args.seed, task=args.task, hz=args.hz, orientation=args.orientation,
                    filter=args.filter, gain=args.gain)


def intrinsics_for(args, width, height):
    if args.intrinsics:
        return Intrinsics.load(args.intrinsics).resized(width, height)
    print("WARNING: approximate intrinsics. Calibrate K and measure palm width before metric experiments.")
    return Intrinsics.approximate(width, height, args.hfov)


def panel(sim, frame, obs, ctrl, pose, reason, mirror, top, elapsed):
    left = draw_hand(frame, obs, mirror)
    left = cv2.resize(left, (640, 480))  # Display only; PnP always sees the original resolution.
    right = sim.render(640, 480, top)
    canvas = np.vstack([np.hstack([left, right]), np.zeros((112, 1280, 3), np.uint8)])
    depth = f"{pose.position[2]:.3f}m" if pose else "N/A"
    lines = [f"{ctrl.state.value} | {ctrl.message}",
             f"Depth {depth} | {reason} | task {sim.cfg.task} | success {sim.success} | wall {elapsed:.1f}s",
             "C neutral | SPACE clutch | X latched stop | R clear latch | V view | Q / ESC exit",
             "Position: away from camera -> +robot X; image right -> -Y; image up -> +Z; pinch -> gripper"]
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (12, 500 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, .49,
                    (230, 230, 230), 1, cv2.LINE_AA)
    cv2.imshow("RGB Gesture Teleoperation", canvas)
    return cv2.waitKey(1) & 255


def step_row(sim, ctrl, packet, pose, command, robot, events, now, reason, truth=None):
    return {"sim_time": sim.time, "active_time": sim.active_time, "control_time": now,
            "capture_time": packet.timestamp if packet else None,
            "frame_id": packet.frame_id if packet else -1,
            "observation": packet.observation if packet else None,
            "pose": pose, "pose_reason": reason, "state": ctrl.state.value,
            "events": events, "command": command, "robot": robot,
            "motor_targets": sim.motor_targets.copy(), "ik_reason": sim.reason,
            "pose_jump_rejections": ctrl.rejected, "workspace_clips": ctrl.clips,
            "capture_to_control_ms": max(0., now - packet.timestamp) * 1000 if packet else None,
            "synthetic_wrist_truth": truth}


def run(args):
    if args.mode == "live" and args.headless:
        raise ValueError("Live control requires a visible window for C/SPACE/X; --headless is not allowed")
    if args.mode == "video" and args.headless and not args.auto_arm:
        raise ValueError("Headless video retargeting requires explicit --auto-arm")
    cfg = config_from_args(args)
    origin = time.monotonic()
    packet = None
    with ExitStack() as stack:
        source = None
        model_hash = None
        if args.mode != "demo":
            model_hash = verify_model(args.model)
        if args.mode == "live":
            source = Webcam(args.camera, args.model, origin, args.width, args.height, cfg.hz)
            stack.callback(source.close)
            while packet is None:
                if source.error:
                    raise RuntimeError(source.error)
                if time.monotonic() - origin > 25:
                    raise TimeoutError("No inference packet within 25 seconds")
                packet = source.poll()
                time.sleep(.01)
        elif args.mode == "video":
            source = Video(args.input, args.model)
            stack.callback(source.close)
            packet = source.poll(0.)
            if packet is None:
                raise ValueError("Input video contains no decodable frame")
        width, height = (packet.frame.shape[1], packet.frame.shape[0]) if packet else (640, 480)
        k = intrinsics_for(args, width, height)
        estimator = PoseEstimator(k, args.palm_width)
        ctrl = Controller(cfg)
        sim = stack.enter_context(Simulation(cfg))
        metadata = {"source": {"demo": "synthetic", "live": "webcam", "video": "video"}[args.mode],
                    "config": asdict(cfg), "intrinsics": asdict(k), "palm_width_m": args.palm_width,
                    "model_sha256": model_hash, "units": "metres, radians, seconds; quaternion xyzw",
                    "camera_to_robot_rotation": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                    "metric_assumption": "fixed camera; measured palm width; calibrated or explicitly approximate K",
                    "timestamp_note": "webcam: host read completion, not sensor exposure; video: CFR frame_index/fps",
                    "input_video_sha256": sha256(args.input) if args.mode == "video" else None,
                    "input_video_fps": source.fps if args.mode == "video" else None,
                    "synthetic_noise_px": args.noise_px if args.mode == "demo" else None,
                    "camera_pixels_saved": bool(args.save_video)}
        rec = Recorder(args.output, metadata, args.save_video, cfg.hz, args.max_mb)
        last_id, pose, pending, top = -1, None, [], False
        rng = np.random.default_rng(cfg.seed)
        auto_armed = False
        end_reason = "duration"
        run_start = time.monotonic()
        next_tick = run_start
        try:
            for tick in range(int(args.seconds * cfg.hz)):
                now = time.monotonic() - origin if args.mode == "live" else tick / cfg.hz
                if args.mode == "live" and time.monotonic() - run_start >= args.seconds:
                    break
                truth = None
                events, pending = pending, []
                if args.mode == "demo":
                    target, pinch = demo_target(cfg.task, tick / cfg.hz, sim.goal(tick / cfg.hz))
                    truth = camera_target(target, cfg.gain)
                    obs = observation(k, truth, now, pinch, args.noise_px, rng, frame_id=tick)
                    packet = Packet(np.zeros((height, width, 3), np.uint8), now, tick, obs, "synthetic")
                elif args.mode == "live":
                    if source.error:
                        ctrl.stop(source.error, emergency=True)
                        end_reason = source.error
                        sim.step(ctrl.step(now, 1. / cfg.hz))
                        raise RuntimeError(source.error)
                    packet = source.poll()
                else:
                    packet = source.poll(now)
                    if source.eof and (packet is None or now - packet.timestamp > 1. / source.fps + 1e-6):
                        end_reason = "video_eof"
                        break
                if packet and packet.frame_id != last_id:
                    if packet.frame.shape[:2] != (height, width):
                        raise ValueError("Input resolution changed within episode")
                    pose = estimator.estimate(packet.observation)
                    ctrl.observe(pose)
                    last_id = packet.frame_id
                if pose is not None and not auto_armed and (args.mode == "demo" or
                                                            (args.mode == "video" and args.auto_arm)):
                    events += ["calibrate", "toggle"]
                    auto_armed = True  # One initial engagement only, never auto-resume after loss.
                state = sim.state()
                for event in events:
                    ctrl.event(event, state.position, state.quaternion, now)
                cmd = ctrl.step(now, 1. / cfg.hz)
                robot = sim.step(cmd)
                if cmd.active and sim.reason != "ok":
                    ctrl.stop(f"IK rejected: {sim.reason}; SPACE to re-anchor")
                reason = estimator.reason if packet and packet.observation else (
                    packet.reason if packet else "awaiting_camera")
                rec.append(step_row(sim, ctrl, packet, pose, cmd, robot, events, now, reason, truth),
                           packet.frame if packet else None)
                if not args.headless:
                    frame = packet.frame if packet else np.zeros((height, width, 3), np.uint8)
                    key = panel(sim, frame, packet.observation if packet else None, ctrl, pose,
                                reason, args.mirror, top, time.monotonic() - run_start)
                    if key in (27, ord('q')):
                        end_reason = "user_exit"
                        break
                    if key == ord('v'):
                        top = not top
                    if key in KEYS:
                        pending.append(KEYS[key])
                if args.mode == "live" or not args.headless:
                    next_tick += 1. / cfg.hz
                    time.sleep(max(0., next_tick - time.monotonic()))
                    if time.monotonic() - next_tick > .25:
                        next_tick = time.monotonic()  # No burst of catch-up robot commands.
            cv2.imwrite(str(Path(args.output) / "preview.png"), sim.render())
        except BaseException as exc:
            end_reason = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            summary = {**sim.task_summary(), "end_reason": end_reason,
                       "wall_runtime_s": time.monotonic() - run_start,
                       "final_controller_state": ctrl.state.value, "pose_jump_rejections": ctrl.rejected,
                       "workspace_clips": ctrl.clips, "validation_kind": metadata["source"]}
            rec.close(summary)
            if not args.headless:
                cv2.destroyAllWindows()
        print(json.dumps(summary, indent=2))
        if args.assert_success and not sim.success:
            raise RuntimeError("Task success assertion failed; inspect episode and diagnostics")
        return summary


def replay(args):
    verify_episode(args.episode)
    meta, rows = read_episode(args.episode)
    cfg = Settings(**meta["config"])
    errors = []
    started = time.monotonic()
    with Simulation(cfg) as sim:
        try:
            for index, row in enumerate(rows):
                if args.mode == "state":
                    sim.restore_visual_state(row["robot"])
                else:
                    state = sim.step(Command(**row["command"]))
                    errors.append(float(np.linalg.norm(state.position - np.array(row["robot"]["position"]))))
                if not args.headless:
                    cv2.imshow("Episode replay", sim.render())
                    if cv2.waitKey(1) & 255 in (27, ord('q')):
                        break
                    time.sleep(max(0., started + (index + 1) / cfg.hz - time.monotonic()))
        finally:
            if not args.headless:
                cv2.destroyAllWindows()
    result = {"mode": args.mode, "rows": len(rows), "max_ee_replay_error_m": max(errors) if errors else None,
              "note": "state replay is visualization, not a dynamics test" if args.mode == "state" else
                      "same recorded Cartesian commands, same physics settings"}
    print(json.dumps(result, indent=2))
    if errors and max(errors) > args.tolerance:
        raise RuntimeError(f"Replay differs by {max(errors):.6g}m; check dependency versions and source revision")
    return result


def reprocess(args):
    """Replay raw landmark observations with one changed filter; no camera/model rerun."""
    verify_episode(args.episode)
    meta, rows = read_episode(args.episode)
    cfg = Settings(**{**meta["config"], "filter": args.filter})
    estimator = PoseEstimator(Intrinsics(**meta["intrinsics"]), meta["palm_width_m"])
    ctrl = Controller(cfg)
    outmeta = {**meta, "source": "landmark_reprocess", "config": asdict(cfg),
               "parent_episode_sha256": sha256(Path(args.episode) / "manifest.json"),
               "camera_pixels_saved": False, "ablation_note": "same observations and original event schedule"}
    # Recorder supplies fresh run provenance / time rather than inheriting the parent's.
    for key in ("created_utc", "provenance", "schema"):
        outmeta.pop(key, None)
    with Simulation(cfg) as sim:
        rec = Recorder(args.output, outmeta)
        last_id, pose = -1, None
        reason = "complete"
        try:
            for row in rows:
                now = row["control_time"]
                obs = Observation(**row["observation"]) if row["observation"] else None
                packet = None
                if row["capture_time"] is not None:
                    packet = Packet(np.zeros((1, 1, 3), np.uint8), row["capture_time"], row["frame_id"], obs,
                                    row["pose_reason"])
                if packet and packet.frame_id != last_id:
                    pose = estimator.estimate(obs)
                    ctrl.observe(pose)
                    last_id = packet.frame_id
                state = sim.state()
                for event in row["events"]:
                    ctrl.event(event, state.position, state.quaternion, now)
                cmd = ctrl.step(now, 1. / cfg.hz)
                robot = sim.step(cmd)
                if cmd.active and sim.reason != "ok":
                    ctrl.stop(f"IK rejected: {sim.reason}")
                rec.append(step_row(sim, ctrl, packet, pose, cmd, robot, row["events"], now,
                                    estimator.reason, row.get("synthetic_wrist_truth")))
        except BaseException as exc:
            reason = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            rec.close({**sim.task_summary(), "end_reason": reason, "validation_kind": "landmark_reprocess"})
    return 0


def evaluate(folder, output, static=False):
    verify_episode(folder)
    meta, rows = read_episode(folder)
    summary = json.loads((Path(folder) / "summary.json").read_text(encoding="utf-8"))
    unique = {r["frame_id"]: r for r in rows if r["frame_id"] >= 0}
    valid = [r for r in unique.values() if r["pose"] is not None]
    active = [r for r in rows if r["command"]["active"]]
    errors = [np.linalg.norm(np.array(r["robot"]["position"]) - np.array(r["command"]["position"]))
              for r in active]
    def quantiles(values):
        values = np.asarray(values, dtype=float)
        return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))} if len(values) else None
    truths = [r for r in valid if r.get("synthetic_wrist_truth") is not None]
    result = {"source": meta["source"], "summary": summary, "rows": len(rows),
              "unique_input_frames": len(unique), "pose_valid_fraction": len(valid) / len(unique) if unique else 0.,
              "command_tracking_rmse_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
              "reprojection_px": quantiles([r["pose"]["reprojection_px"] for r in valid]),
              "model_inference_ms": quantiles([r["observation"]["inference_ms"] for r in valid]),
              "host_capture_to_control_ms": quantiles([r["capture_to_control_ms"] for r in rows
                                                       if r["capture_to_control_ms"] is not None]),
              "synthetic_pose_rmse_m": float(np.sqrt(np.mean([
                  np.sum((np.array(r["pose"]["position"]) - r["synthetic_wrist_truth"]) ** 2) for r in truths])))
                  if truths else None,
              "stationary_wrist_std_m": np.std([r["pose"]["position"] for r in valid], axis=0).tolist()
                  if static and valid else None,
              "metric_notes": ["Reprojection residual is not 3-D accuracy.",
                               "Command tracking measures simulation following, not hand-pose accuracy.",
                               "Host read completion excludes exposure, USB, and monitor latency.",
                               "Static jitter is meaningful only when the hand was deliberately held still."]}
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    output.mkdir(parents=True)
    dump(output / "metrics.json", result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    times = np.array([r["sim_time"] for r in rows])
    target = np.array([r["command"]["position"] for r in rows])
    actual = np.array([r["robot"]["position"] for r in rows])
    for axis, name in enumerate(("x", "y", "z")):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(times, target[:, axis], label="command")
        ax.plot(times, actual[:, axis], label="robot")
        ax.set(xlabel="Simulation time (s)", ylabel=f"Robot {name} (m)", title=f"{name.upper()} tracking")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"tracking_{name}.png", dpi=160)
        plt.close(fig)
    print(json.dumps(result, indent=2))
    return result
