"""Reprocess identical recorded landmarks under both filters; keep failures and provenance."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from gesture.runtime import reprocess, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for method in ("none", "one-euro"):
        episode = root / method
        reprocess(SimpleNamespace(episode=args.episode, output=episode, filter=method))
        metrics = evaluate(episode, root / f"{method}-metrics")
        summary = metrics["summary"]
        rows.append({"filter": method, "success": summary["success"],
                     "active_seconds": summary["active_seconds"],
                     "path_rmse_m": summary["path_rmse_m"],
                     "ik_rejections": summary["ik_rejections"],
                     "command_tracking_rmse_m": metrics["command_tracking_rmse_m"],
                     "pose_valid_fraction": metrics["pose_valid_fraction"]})
    with (root / "comparison.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {root / 'comparison.csv'}. This is open-loop replay, not a user-study result.")


if __name__ == "__main__":
    main()
