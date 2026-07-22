#!/usr/bin/env python3
"""Convert and evaluate every refined scenario with LMDrive."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from convert_dynamic_to_lmdrive_xml import convert


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/refined"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "2010")))
    parser.add_argument("--tm-port", type=int, default=int(os.environ.get("TM_PORT", "8010")))
    parser.add_argument("--only", action="append", default=[], help="Run only the named scene folder")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    refined_root = (repo_root / args.root).resolve() if not args.root.is_absolute() else args.root
    launcher = repo_root / "tools" / "run_lmdrive_dynamic_scenario.sh"
    scene_dirs = sorted(
        path for path in refined_root.iterdir() if path.is_dir() and (path / "dynamic").is_dir()
    )
    if args.only:
        requested = set(args.only)
        scene_dirs = [path for path in scene_dirs if path.name in requested]

    summary = {
        "schema_version": "lmdrive-refined-batch-v1",
        "root": str(refined_root),
        "port": args.port,
        "traffic_manager_port": args.tm_port,
        "started_at_unix": time.time(),
        "scenes": [],
    }
    summary_path = refined_root / "lmdrive_batch_summary.json"

    for index, scene_dir in enumerate(scene_dirs, 1):
        dynamic_dir = scene_dir / "dynamic"
        xml_path = dynamic_dir / "s0000_c0_lmdrive_scenario.xml"
        metrics_path = dynamic_dir / "s0000_c0_lmdrive_metrics.json"
        log_path = dynamic_dir / "s0000_c0_lmdrive_run.log"
        record = {
            "scene_folder": scene_dir.name,
            "xml": str(xml_path),
            "metrics": str(metrics_path),
            "log": str(log_path),
            "status": "pending",
        }
        summary["scenes"].append(record)
        print("\n[{}/{}] {}".format(index, len(scene_dirs), scene_dir.name), flush=True)
        try:
            convert(dynamic_dir, xml_path)
        except Exception as error:  # continue the complete requested batch
            record.update(status="conversion_failed", error="{}: {}".format(type(error).__name__, error))
            print("conversion failed: {}".format(record["error"]), flush=True)
            _write_json(summary_path, summary)
            continue

        env = os.environ.copy()
        env.update(PORT=str(args.port), TM_PORT=str(args.tm_port), PYTHONUNBUFFERED="1")
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [str(launcher), str(xml_path)],
                cwd=str(repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return_code = process.wait()
        record["return_code"] = return_code
        record["wall_time_s"] = round(time.time() - started, 3)
        if metrics_path.is_file():
            with metrics_path.open(encoding="utf-8") as stream:
                metrics = json.load(stream)
            leaderboard = metrics.get("leaderboard") or {}
            record.update(
                status="evaluated" if return_code == 0 else "run_failed_with_metrics",
                scores=leaderboard.get("scores"),
                leaderboard_status=leaderboard.get("status"),
                collision=metrics.get("collision"),
                min_center_distance_m=metrics.get("min_center_distance_m"),
                min_ttc_s=metrics.get("min_ttc_s"),
            )
        else:
            record["status"] = "run_failed"
        _write_json(summary_path, summary)

    summary["finished_at_unix"] = time.time()
    summary["counts"] = {
        status: sum(item["status"] == status for item in summary["scenes"])
        for status in sorted({item["status"] for item in summary["scenes"]})
    }
    _write_json(summary_path, summary)
    print("\nBatch summary: {}".format(summary_path), flush=True)
    return 0 if all(item["status"] == "evaluated" for item in summary["scenes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
