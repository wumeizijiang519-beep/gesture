"""Explicit commands. Run `python -m gesture --help` without opening devices."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import traceback


def positive(value):
    import math
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def parser():
    root = argparse.ArgumentParser(description="Task 13: RGB hand -> metric pose -> simulated Panda")
    root.add_argument("--debug", action="store_true", help="Print full traceback on failure")
    commands = root.add_subparsers(dest="command", required=True)
    dl = commands.add_parser("download-model", help="Download and SHA256-verify the versioned official model")
    dl.add_argument("--output", default="models/hand_landmarker.task")
    doctor = commands.add_parser("doctor", help="Dependency, simulator, model, optional camera checks")
    doctor.add_argument("--model", default="models/hand_landmarker.task")
    doctor.add_argument("--camera", type=int, default=None, help="Opt in to camera read test")
    for mode in ("demo", "live", "video"):
        sub = commands.add_parser(mode, help={"demo": "Synthetic full geometry/control integration run",
                                             "live": "Laptop camera teleoperation", "video": "Offline RGB retargeting"}[mode])
        sub.set_defaults(mode=mode)
        sub.add_argument("--task", choices=["reach", "path", "pick-place"], default="reach")
        sub.add_argument("--output", required=True, help="New episode folder; never overwritten")
        sub.add_argument("--seconds", type=positive, default=30. if mode == "demo" else 300.)
        sub.add_argument("--seed", type=int, default=7)
        sub.add_argument("--hz", type=int, choices=[20, 24, 30, 40, 60], default=30)
        sub.add_argument("--filter", choices=["one-euro", "none"], default="one-euro")
        sub.add_argument("--gain", type=positive, default=1.)
        sub.add_argument("--orientation", action="store_true", help="Also map relative wrist orientation (experimental)")
        sub.add_argument("--headless", action="store_true")
        sub.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True,
                         help="Display mirror only; never change geometry")
        sub.add_argument("--intrinsics", help="Camera calibration JSON; omitted = explicitly approximate K")
        sub.add_argument("--palm-width", type=positive, default=.08, help="Distance between MCP 5 and 17, metres")
        sub.add_argument("--hfov", type=positive, default=65., help="Approximate horizontal FOV if K is unavailable")
        sub.add_argument("--save-video", action="store_true", help="Opt in to local RGB storage")
        sub.add_argument("--max-mb", type=positive, default=512.)
        sub.add_argument("--model", default="models/hand_landmarker.task")
        sub.add_argument("--assert-success", action="store_true", help="Nonzero exit code when task fails")
        sub.add_argument("--noise-px", type=float, default=0., help="Synthetic-only Gaussian pixel noise")
        sub.add_argument("--auto-arm", action="store_true", help="Video only: explicitly engage once at first valid pose")
        if mode == "live":
            sub.add_argument("--camera", type=int, default=0)
            sub.add_argument("--width", type=int, default=640)
            sub.add_argument("--height", type=int, default=480)
        if mode == "video":
            sub.add_argument("--input", required=True)
    replay = commands.add_parser("replay", help="Integrity-checked command dynamics replay or exact state display")
    replay.add_argument("--episode", required=True)
    replay.add_argument("--mode", choices=["command", "state"], default="command")
    replay.add_argument("--headless", action="store_true")
    replay.add_argument("--tolerance", type=positive, default=1e-5)
    for name in ("export", "evaluate", "reprocess"):
        sub = commands.add_parser(name)
        sub.add_argument("--episode", required=True)
        sub.add_argument("--output", required=True)
        if name == "evaluate":
            sub.add_argument("--static", action="store_true", help="Only for deliberately stationary-hand recordings")
        if name == "reprocess":
            sub.add_argument("--filter", choices=["one-euro", "none"], required=True)
    verify = commands.add_parser("verify-data", help="Check all recorded files against the episode manifest")
    verify.add_argument("--episode", required=True)
    for name in ("board", "calibrate"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", default="calibration/checkerboard.svg" if name == "board" else "calibration/camera.json")
        sub.add_argument("--cols", type=int, default=9, help="Inner corners, not squares")
        sub.add_argument("--rows", type=int, default=6)
        sub.add_argument("--square-mm", type=positive, default=20.)
        if name == "calibrate":
            sub.add_argument("--camera", type=int, default=0)
            sub.add_argument("--width", type=int, default=640)
            sub.add_argument("--height", type=int, default=480)
    return root


def doctor(args):
    import importlib.metadata
    from .control import Settings, Command
    from .simulation import Simulation
    from .vision import verify_model, Detector, open_camera
    checks = {"python": sys.version.split()[0], "platform": platform.platform(), "checks": {}}
    failed = False
    for name in ("numpy", "scipy", "opencv-contrib-python", "pybullet", "mediapipe"):
        try:
            checks["checks"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            checks["checks"][name] = "MISSING"
            failed = True
    try:
        with Simulation(Settings()) as sim:
            state = sim.state()
            sim.step(Command(state.position, state.quaternion, .08, False))
            checks["checks"]["simulator"] = {"joints": len(sim.joints), "render_shape": list(sim.render().shape)}
    except Exception as exc:
        checks["checks"]["simulator"] = f"FAILED: {exc}"
        failed = True
    try:
        import numpy as np
        checks["checks"]["model_sha256"] = verify_model(args.model)
        detector = Detector(args.model)
        try:
            result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8), 0., 0)
            checks["checks"]["blank_frame_no_hand"] = result is None
            failed |= result is not None
        finally:
            detector.close()
    except Exception as exc:
        checks["checks"]["model"] = f"FAILED: {exc}"
        failed = True
    if args.camera is not None:
        try:
            cap = open_camera(args.camera)
            try:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Read failed")
                checks["checks"]["camera"] = {"shape": list(frame.shape), "note": "read only; no pixels saved"}
            finally:
                cap.release()
        except Exception as exc:
            checks["checks"]["camera"] = f"FAILED: {exc}"
            failed = True
    print(json.dumps(checks, indent=2))
    return int(failed)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in ("live", "calibrate") or (args.command in ("demo", "video", "replay") and not args.headless):
            if platform.system() == "Linux" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                raise RuntimeError("No desktop display. Use --headless for demo/replay/video, not live/calibrate")
        if args.command in ("demo", "live", "video"):
            import math
            if not math.isfinite(args.noise_px) or args.noise_px < 0:
                raise ValueError("noise-px must be finite and nonnegative")
            if args.seconds > 3600:
                raise ValueError("One episode is capped at one hour; start a new episode")
            if args.mode == "live" and args.auto_arm:
                raise ValueError("Live automatic arming is disabled; use C then SPACE")
            from .runtime import run
            run(args)
        elif args.command == "download-model":
            from .vision import download_model
            print(download_model(args.output))
        elif args.command == "doctor":
            return doctor(args)
        elif args.command == "replay":
            from .runtime import replay
            replay(args)
        elif args.command == "reprocess":
            from .runtime import reprocess
            reprocess(args)
        elif args.command == "evaluate":
            from .runtime import evaluate
            evaluate(args.episode, args.output, args.static)
        elif args.command == "export":
            from .data import export_npz
            export_npz(args.episode, args.output)
            print(f"Wrote {Path(args.output)} (load with allow_pickle=False)")
        elif args.command == "verify-data":
            from .data import verify_episode
            verify_episode(args.episode)
            print("Episode checksums verified")
        elif args.command == "board":
            from .calibration import make_board
            make_board(args.output, args.cols, args.rows, args.square_mm)
            print(f"Wrote {args.output}; print at 100% and check square size with a ruler")
        elif args.command == "calibrate":
            from .calibration import calibrate_camera
            return calibrate_camera(args)
        return 0
    except KeyboardInterrupt:
        print("Interrupted; completed records retained.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
