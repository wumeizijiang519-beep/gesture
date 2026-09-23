"""Printable metric checkerboard and OpenCV camera calibration (no hidden K defaults)."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from .geometry import Intrinsics
from .vision import open_camera
from .data import dump


def make_board(output, cols=9, rows=6, square_mm=20.):
    if min(cols, rows) < 3 or square_mm <= 0:
        raise ValueError("At least 3x3 inner corners and positive square size required")
    width, height = (cols + 1) * square_mm, (rows + 1) * square_mm
    if width > 277 or height > 190:
        raise ValueError("Board does not fit landscape A4 with margins")
    x0, y0 = (297 - width) / 2, (210 - height) / 2
    shapes = ['<rect width="297" height="210" fill="white"/>']
    for row in range(rows + 1):
        for col in range(cols + 1):
            if (row + col) % 2 == 0:
                shapes.append(f'<rect x="{x0 + col * square_mm}" y="{y0 + row * square_mm}" '
                              f'width="{square_mm}" height="{square_mm}" fill="black"/>')
    shapes.append(f'<text x="10" y="204" font-size="3">{cols} x {rows} inner corners; '
                  f'{square_mm} mm squares. Print at 100%, not fit-to-page; verify with a ruler.</text>')
    path = Path(output)
    if path.suffix.lower() != ".svg":
        raise ValueError("Board output must end in .svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="297mm" height="210mm" '
                    'viewBox="0 0 297 210">' + ''.join(shapes) + '</svg>', encoding="utf-8")


def solve_calibration(corners, size, cols=9, rows=6, square_mm=20.):
    if len(corners) < 12:
        raise ValueError("Collect at least 12 diverse checkerboard views")
    board = np.zeros((cols * rows, 3), dtype=np.float32)
    board[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm / 1000.
    observations = [np.asarray(c, dtype=np.float32).reshape(-1, 1, 2) for c in corners]
    if any(c.shape[0] != cols * rows or not np.isfinite(c).all() for c in observations):
        raise ValueError("Invalid checkerboard corner array")
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera([board] * len(corners), observations,
                                                              size, None, None)
    errors = []
    for detected, rv, tv in zip(observations, rvecs, tvecs):
        projected = cv2.projectPoints(board, rv, tv, matrix, distortion)[0]
        errors.append(float(np.sqrt(np.mean(np.sum((detected - projected) ** 2, axis=2)))))
    if not np.isfinite(matrix).all() or rms > 1.5 or max(errors) > 3.:
        raise ValueError(f"Poor calibration (RMS {rms:.3f}px); collect sharper, diverse, tilted views")
    if not .2 * size[0] < matrix[0, 0] < 4 * size[0] or not .2 * size[1] < matrix[1, 1] < 6 * size[1]:
        raise ValueError("Implausible focal length; check checkerboard size and view diversity")
    k = Intrinsics(size[0], size[1], matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2],
                   tuple(distortion.ravel()), True)
    return k, {"rms_px": rms, "per_view_rms_px": errors, "views": len(corners),
               "inner_corners": [cols, rows], "square_mm": square_mm}


def calibrate_camera(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Calibration exists: {output}; choose a new file")
    cap = open_camera(args.camera, args.width, args.height)
    corners, signatures = [], []
    last_size = None
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                raise RuntimeError("Camera read failed")
            size = (image.shape[1], image.shape[0])
            if last_size is not None and last_size != size:
                raise ValueError("Camera resolution changed during calibration")
            last_size = size
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            found, points = cv2.findChessboardCornersSB(gray, (args.cols, args.rows))
            display = image.copy()
            if found:
                cv2.drawChessboardCorners(display, (args.cols, args.rows), points, True)
            cv2.putText(display, f"Views {len(corners)} / >=12 | SPACE capture | ENTER solve | ESC quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (40, 200, 60), 1, cv2.LINE_AA)
            cv2.imshow("Camera calibration", display)
            key = cv2.waitKey(1) & 255
            if key == 27:
                return 0
            if key == 32 and found:
                signature = points.reshape(-1, 2) / np.array(size)
                if signatures and min(np.sqrt(np.mean((signature - s) ** 2)) for s in signatures) < .025:
                    print("View too similar. Move/tilt the board or change its distance.")
                else:
                    corners.append(points.copy())
                    signatures.append(signature)
                    print(f"Captured view {len(corners)}")
            if key in (10, 13):
                try:
                    k, report = solve_calibration(corners, size, args.cols, args.rows, args.square_mm)
                except ValueError as exc:
                    print(exc)
                    continue
                k.save(output)
                dump(output.with_suffix(".report.json"), report)
                np.savez_compressed(output.with_suffix(".corners.npz"), corners=np.array(corners), image_size=size)
                print(json.dumps(report, indent=2))
                return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()
