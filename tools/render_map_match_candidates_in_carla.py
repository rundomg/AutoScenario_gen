"""Render accepted map-match candidates with CARLA RGB cameras.

The map-match report contains compact per-world summaries, so this utility
replays the scorer against ``data/map_cache`` to recover the selected location
in every accepted world.  It then loads each CARLA map and captures a real
top-down RGB sensor image at that location.

The CARLA server must already be running.
"""

from __future__ import annotations

import argparse
import json
import queue
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tools.map_matcher_v2 import score_candidate_v2
from tools.scene_map_matcher import SceneMapMatcher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--signature", required=True, type=Path)
    parser.add_argument("--cache-dir", default=Path("data/map_cache"), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--timeout", default=120.0, type=float)
    parser.add_argument("--height", default=80.0, type=float)
    parser.add_argument("--fov", default=55.0, type=float)
    parser.add_argument("--size", default=1024, type=int)
    parser.add_argument(
        "--map-settle-seconds",
        default=10.0,
        type=float,
        help="Wait after load_world so large-map assets can finish streaming.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip candidates whose non-empty output image already exists.",
    )
    parser.add_argument(
        "--only-world",
        help="Render only a matching world name, e.g. Town13.",
    )
    parser.add_argument(
        "--skip-world",
        action="append",
        default=[],
        help="Skip a matching world name; may be supplied more than once.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Try the next map after a failure (unsafe if the CARLA server timed out).",
    )
    parser.add_argument(
        "--warmup-ticks",
        default=10,
        type=int,
        help="Frames to wait after loading each map before saving an image.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_world_name(world: str) -> str:
    return world.replace("/", "_").replace("\\", "_")


def _recover_best_candidates(
    report: Dict[str, Any], signature_path: Path, cache_dir: Path
) -> List[Dict[str, Any]]:
    signature = _load_json(signature_path)
    recovered: List[Dict[str, Any]] = []
    for summary in report.get("accepted") or []:
        world_name = str(summary["world"])
        cache_path = cache_dir / f"{_safe_world_name(world_name)}.json"
        cache_candidates = list((_load_json(cache_path).get("candidates") or []))
        SceneMapMatcher._normalize_cached_parking_sides(cache_candidates)
        best_candidate = None
        best_score = -1.0
        for candidate in cache_candidates:
            score, details = score_candidate_v2(signature, candidate)
            if not details.get("hard_reject") and score > best_score:
                best_candidate = candidate
                best_score = float(score)
        if best_candidate is not None:
            recovered.append(
                {
                    "world": world_name,
                    "score": best_score,
                    "candidate": best_candidate,
                }
            )
    return recovered


def _map_name_variants(world_name: str) -> Iterable[str]:
    raw = str(world_name).strip()
    short = raw.replace("Carla/Maps/", "")
    leaf = short.split("/")[-1]
    # Large maps are reported as Carla/Maps/Town13/Town13 by the cache, while
    # load_world generally accepts the leaf name (Town13) most reliably.
    if "/" in short:
        variants = (leaf, raw, short, f"Carla/Maps/{leaf}")
    else:
        variants = (raw, short, leaf, f"Carla/Maps/{short}", f"Carla/Maps/{leaf}")
    seen = set()
    for value in variants:
        if value and value not in seen:
            seen.add(value)
            yield value


def _load_world(client: Any, world_name: str) -> Any:
    last_error: Exception | None = None
    for variant in _map_name_variants(world_name):
        try:
            print(f"  loading {variant}")
            return client.load_world(variant)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"could not load {world_name}: {last_error}")


def _advance_world(world: Any) -> None:
    settings = world.get_settings()
    if bool(settings.synchronous_mode):
        world.tick()
    else:
        world.wait_for_tick()


def _capture(
    carla: Any,
    world: Any,
    candidate: Dict[str, Any],
    output_path: Path,
    *,
    height: float,
    fov: float,
    size: int,
    warmup_ticks: int,
) -> None:
    location = candidate.get("location") or {}
    x = float(location["x"])
    y = float(location["y"])
    z = float(location.get("z") or 0.0)
    yaw = float(candidate.get("yaw") or 0.0)

    # Snap the cached position onto the loaded CARLA road to obtain its real Z.
    waypoint = world.get_map().get_waypoint(
        carla.Location(x=x, y=y, z=z), project_to_road=True
    )
    if waypoint is not None:
        z = float(waypoint.transform.location.z)

    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(size))
    blueprint.set_attribute("image_size_y", str(size))
    blueprint.set_attribute("fov", str(fov))
    blueprint.set_attribute("sensor_tick", "0.0")
    transform = carla.Transform(
        carla.Location(x=x, y=y, z=z + height),
        carla.Rotation(pitch=-90.0, yaw=yaw, roll=0.0),
    )

    images: queue.Queue[Any] = queue.Queue()
    sensor = world.spawn_actor(blueprint, transform)
    listening = False
    try:
        sensor.listen(images.put)
        listening = True
        latest = None
        for _ in range(max(1, warmup_ticks)):
            _advance_world(world)
            try:
                latest = images.get(timeout=10.0)
            except queue.Empty:
                pass
        if latest is None:
            raise RuntimeError("RGB camera produced no image")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        latest.save_to_disk(str(output_path))
    finally:
        if listening:
            try:
                sensor.stop()
            except Exception as exc:
                print(f"  warning: failed to stop camera: {exc}")
        try:
            sensor.destroy()
        except Exception as exc:
            # Preserve the original capture error. A destroy timeout normally
            # means the server is already unavailable and must be restarted.
            print(f"  warning: failed to destroy camera: {exc}")


def _write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    try:
        import carla  # type: ignore
    except ImportError as exc:
        raise RuntimeError("CARLA Python API is not installed in this environment") from exc

    report = _load_json(args.candidates)
    entries = sorted(
        _recover_best_candidates(report, args.signature, args.cache_dir),
        key=lambda row: row["score"],
        reverse=True,
    )
    if not entries:
        raise RuntimeError("No accepted candidates could be recovered from the map cache")
    for original_rank, entry in enumerate(entries, 1):
        entry["rank"] = original_rank
    if args.only_world:
        token = args.only_world.lower()
        entries = [entry for entry in entries if token in str(entry["world"]).lower()]
        if not entries:
            raise RuntimeError(f"No accepted world matches --only-world {args.only_world!r}")
    for skipped_world in args.skip_world:
        token = skipped_world.lower()
        entries = [entry for entry in entries if token not in str(entry["world"]).lower()]
    if not entries:
        raise RuntimeError("No candidates remain after applying world filters")

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    manifest_path = args.output_dir / "carla_bev_manifest.json"

    for position, entry in enumerate(entries, 1):
        rank = int(entry["rank"])
        world_name = str(entry["world"])
        short_name = world_name.split("/")[-1].replace("_Opt", "")
        output_path = args.output_dir / f"{rank:02d}_{short_name}_carla_bev.png"
        row: Dict[str, Any] = {
            "rank": rank,
            "world": world_name,
            "score": entry["score"],
            "location": entry["candidate"].get("location"),
            "yaw": entry["candidate"].get("yaw"),
            "image_path": str(output_path.resolve()),
        }
        print(f"[{position}/{len(entries)}; rank {rank}] {world_name}")
        if args.resume and output_path.is_file() and output_path.stat().st_size > 0:
            row["status"] = "skipped_existing"
            manifest.append(row)
            _write_manifest(manifest_path, manifest)
            print(f"  skipped existing {output_path}")
            continue
        try:
            world = _load_world(client, world_name)
            # Give streamed map assets a moment to settle before creating the sensor.
            print(f"  waiting {args.map_settle_seconds:g}s for map assets")
            time.sleep(max(0.0, args.map_settle_seconds))
            _advance_world(world)
            _capture(
                carla,
                world,
                entry["candidate"],
                output_path,
                height=args.height,
                fov=args.fov,
                size=args.size,
                warmup_ticks=args.warmup_ticks,
            )
            row["status"] = "rendered"
            print(f"  saved {output_path}")
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            print(f"  FAILED: {exc}")
        manifest.append(row)
        _write_manifest(manifest_path, manifest)
        if row["status"] == "failed" and not args.continue_on_error:
            print("Stopping because the CARLA connection may be unusable; restart the server before retrying.")
            break

    _write_manifest(manifest_path, manifest)
    rendered = sum(row["status"] == "rendered" for row in manifest)
    completed = sum(
        row["status"] in {"rendered", "skipped_existing"} for row in manifest
    )
    print(f"Rendered {rendered}; completed or previously existing {completed}/{len(manifest)}")
    print(f"Manifest: {manifest_path}")
    return 0 if completed == len(manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
