"""Render vehicle-free top-down previews for cached map-match candidates.

The map-match debug JSON intentionally stores compact per-world summaries.  This
utility replays the same v2 scorer against ``data/map_cache`` to recover each
accepted world's best candidate, then draws nearby cached lane segments without
loading CARLA or spawning actors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image, ImageDraw, ImageFont

from tools.map_matcher_v2 import score_candidate_v2
from tools.scene_map_matcher import SceneMapMatcher


Point = Tuple[float, float]


def _safe_world_name(world: str) -> str:
    return world.replace("/", "_").replace("\\", "_")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _recover_best_candidates(
    candidates_report: Dict[str, Any], cache_dir: Path
) -> List[Dict[str, Any]]:
    accepted = candidates_report.get("accepted") or []
    signature_path = Path(candidates_report["_signature_path"])
    signature = _load_json(signature_path)
    recovered: List[Dict[str, Any]] = []

    for summary in accepted:
        world = str(summary["world"])
        cache_path = cache_dir / f"{_safe_world_name(world)}.json"
        cache = _load_json(cache_path)
        candidates = list(cache.get("candidates") or [])
        SceneMapMatcher._normalize_cached_parking_sides(candidates)

        best_candidate: Dict[str, Any] | None = None
        best_score = -1.0
        best_details: Dict[str, Any] = {}
        for candidate in candidates:
            score, details = score_candidate_v2(signature, candidate)
            if not details.get("hard_reject") and score > best_score:
                best_candidate = candidate
                best_score = float(score)
                best_details = details
        if best_candidate is None:
            continue
        recovered.append(
            {
                "world": world,
                "score": best_score,
                "summary_score": float(summary.get("score") or 0.0),
                "candidate": best_candidate,
                "score_details": best_details,
                "cache_path": str(cache_path),
                "cache_candidates": candidates,
            }
        )
    return recovered


def _transform(point: Point, origin: Point, yaw_deg: float) -> Point:
    dx, dy = point[0] - origin[0], point[1] - origin[1]
    angle = math.radians(yaw_deg)
    # Screen y follows the candidate's forward direction; x is lateral.
    lateral = -math.sin(angle) * dx + math.cos(angle) * dy
    forward = math.cos(angle) * dx + math.sin(angle) * dy
    return lateral, forward


def _nearby_segments(
    candidates: Iterable[Dict[str, Any]],
    origin: Point,
    yaw_deg: float,
    radius_m: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        lane = candidate.get("candidate_lane") or {}
        start = lane.get("start") or {}
        end = lane.get("end") or {}
        if not all(key in start and key in end for key in ("x", "y")):
            continue
        midpoint = (
            (float(start["x"]) + float(end["x"])) / 2.0,
            (float(start["y"]) + float(end["y"])) / 2.0,
        )
        if math.dist(midpoint, origin) > radius_m * 1.45:
            continue
        p1 = _transform((float(start["x"]), float(start["y"])), origin, yaw_deg)
        p2 = _transform((float(end["x"]), float(end["y"])), origin, yaw_deg)
        if max(abs(p1[0]), abs(p1[1]), abs(p2[0]), abs(p2[1])) > radius_m * 1.75:
            continue
        rows.append(
            {
                "points": [p1, p2],
                "road_id": lane.get("road_id"),
                "lane_id": lane.get("lane_id"),
                "is_junction": bool(start.get("is_junction") or end.get("is_junction")),
            }
        )
    return rows


def _render_one(entry: Dict[str, Any], output_path: Path, radius_m: float) -> None:
    candidate = entry["candidate"]
    location = candidate.get("location") or {}
    origin = (float(location["x"]), float(location["y"]))
    yaw = float(candidate.get("yaw") or 0.0)
    anchor_road = (candidate.get("candidate_lane") or {}).get("road_id")
    segments = _nearby_segments(entry["cache_candidates"], origin, yaw, radius_m)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    fig.patch.set_facecolor("#F3F6F1")
    ax.set_facecolor("#DDE9D7")

    all_lines = [row["points"] for row in segments]
    if all_lines:
        ax.add_collection(
            LineCollection(all_lines, colors="#515B66", linewidths=16, capstyle="round", zorder=1)
        )
        ax.add_collection(
            LineCollection(all_lines, colors="#AEB7C0", linewidths=12, capstyle="round", zorder=2)
        )

    junction_lines = [row["points"] for row in segments if row["is_junction"]]
    if junction_lines:
        ax.add_collection(
            LineCollection(junction_lines, colors="#87939E", linewidths=13, capstyle="round", zorder=3)
        )

    anchor_lines = [row["points"] for row in segments if row["road_id"] == anchor_road]
    if anchor_lines:
        ax.add_collection(
            LineCollection(anchor_lines, colors="#374653", linewidths=10, capstyle="round", zorder=4)
        )
        ax.add_collection(
            LineCollection(
                anchor_lines,
                colors="#E8EEF3",
                linewidths=1.2,
                linestyles="dashed",
                zorder=5,
            )
        )

    # Candidate anchor is a map-location marker, not a spawned vehicle.
    selected = entry["world"].endswith("Town03_Opt")
    accent = "#19C3A3" if selected else "#3B82F6"
    ax.scatter([0], [0], s=165, color="#FFFFFF", edgecolor=accent, linewidth=3, zorder=8)
    ax.scatter([0], [0], s=38, color=accent, zorder=9)
    ax.arrow(0, 4, 0, 12, width=0.45, head_width=3.2, head_length=4.2,
             color=accent, length_includes_head=True, zorder=8)

    has_left = bool(candidate.get("left_parking_lane_present"))
    has_right = bool(candidate.get("right_parking_lane_present"))
    topology = str(candidate.get("candidate_topology_type") or "unknown")
    short_world = entry["world"].split("/")[-1].replace("_Opt", "")
    badge = "SELECTED" if selected else "CANDIDATE"
    ax.text(
        0.035, 0.955, short_world, transform=ax.transAxes, ha="left", va="top",
        fontsize=18, fontweight="bold", color="#14233B",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#D7E1EA", alpha=0.96),
        zorder=12,
    )
    ax.text(
        0.965, 0.955, f"{entry['score']:.3f}\n{badge}", transform=ax.transAxes,
        ha="right", va="top", fontsize=10, fontweight="bold", color="#FFFFFF",
        linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=accent, edgecolor="none", alpha=0.96),
        zorder=12,
    )
    parking = "parking L+R" if has_left and has_right else "no bilateral parking"
    ax.text(
        0.035, 0.045, f"{topology.replace('_', ' ')}  ·  {parking}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5,
        color="#334155", bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="none", alpha=0.9),
        zorder=12,
    )
    ax.text(0.965, 0.045, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#475569", zorder=12)
    ax.arrow(0.965, 0.085, 0, 0.045, transform=ax.transAxes, width=0.003,
             head_width=0.02, head_length=0.018, color="#475569", zorder=12)

    ax.set_xlim(-radius_m, radius_m)
    ax.set_ylim(-radius_m, radius_m)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches=None, pad_inches=0)
    plt.close(fig)


def _make_contact_sheet(image_paths: List[Path], output_path: Path) -> None:
    tile_size = 278
    gap = 22
    margin_x = 61
    first_row_y = 150
    second_row_y = 472
    width = 1600
    height = 900
    canvas = Image.new("RGB", (width, height), "#F4F7FA")
    draw = ImageDraw.Draw(canvas)
    font_root = Path("/usr/share/fonts/truetype/dejavu")
    try:
        title_font = ImageFont.truetype(str(font_root / "DejaVuSans-Bold.ttf"), 38)
        subtitle_font = ImageFont.truetype(str(font_root / "DejaVuSans.ttf"), 19)
        footer_font = ImageFont.truetype(str(font_root / "DejaVuSans.ttf"), 15)
    except OSError:
        title_font = subtitle_font = footer_font = ImageFont.load_default()
    draw.text((margin_x, 38), "Map Matching Candidates", fill="#14233B", font=title_font)
    draw.text(
        (margin_x, 92),
        "Vehicle-free top-down previews  |  cached CARLA lane topology  |  Town03 selected",
        fill="#64748B",
        font=subtitle_font,
    )
    for index, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        if index < 5:
            col = index
            x = margin_x + col * (tile_size + gap)
            y = first_row_y
        else:
            col = index - 5
            x = margin_x + (tile_size + gap) // 2 + col * (tile_size + gap)
            y = second_row_y
        canvas.paste(image, (x, y))
    draw.line((margin_x, 805, width - margin_x, 805), fill="#D7E1EA", width=2)
    draw.text(
        (margin_x, 830),
        "Marker = candidate anchor and forward direction; no vehicles are spawned or rendered.",
        fill="#64748B",
        font=footer_font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--signature", required=True, type=Path)
    parser.add_argument("--cache-dir", default=Path("data/map_cache"), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--radius", default=65.0, type=float)
    args = parser.parse_args()

    report = _load_json(args.candidates)
    report["_signature_path"] = str(args.signature)
    recovered = _recover_best_candidates(report, args.cache_dir)
    image_paths: List[Path] = []
    manifest: List[Dict[str, Any]] = []
    for rank, entry in enumerate(sorted(recovered, key=lambda row: row["score"], reverse=True), 1):
        short_world = entry["world"].split("/")[-1].replace("_Opt", "")
        output_path = args.output_dir / f"{rank:02d}_{short_world}_topdown.png"
        _render_one(entry, output_path, args.radius)
        image_paths.append(output_path)
        candidate = entry["candidate"]
        manifest.append(
            {
                "rank": rank,
                "world": entry["world"],
                "score": round(entry["score"], 6),
                "summary_score": round(entry["summary_score"], 6),
                "location": candidate.get("location"),
                "yaw": candidate.get("yaw"),
                "candidate_lane": candidate.get("candidate_lane"),
                "topology": candidate.get("candidate_topology_type"),
                "left_parking_lane_present": candidate.get("left_parking_lane_present"),
                "right_parking_lane_present": candidate.get("right_parking_lane_present"),
                "image_path": str(output_path.resolve()),
            }
        )

    sheet_path = args.output_dir / "s0000_c0_map_candidates_topdown.png"
    _make_contact_sheet(image_paths, sheet_path)
    (args.output_dir / "candidate_topdown_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Rendered {len(image_paths)} candidates to {args.output_dir}")
    print(f"Contact sheet: {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
