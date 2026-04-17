"""Top-down 2D render of a Road DSL plan_view.

Usage::

    from opendrive_experiment.tools.dsl_visualizer import render_dsl_topdown
    png_path = render_dsl_topdown(dsl, "/tmp/road_layout.png")

The output PNG can be passed to a VLM alongside the original traffic image to
perform semantic closed-loop validation ("does this road layout match the scene?").
"""
import math
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _line_points(x: float, y: float, hdg: float, length: float, n: int = 2) -> List[Tuple[float, float]]:
    """Sample n points along a straight line segment."""
    ts = np.linspace(0.0, length, n)
    return [(x + math.cos(hdg) * t, y + math.sin(hdg) * t) for t in ts]


def _arc_points(
    x: float, y: float, hdg: float, length: float, curvature: float, n: int = 40
) -> List[Tuple[float, float]]:
    """Sample n points along a constant-curvature arc segment.

    OpenDRIVE convention: positive curvature = left turn (CCW), negative = right turn (CW).
    """
    if abs(curvature) < 1e-9:
        return _line_points(x, y, hdg, length, n)
    r = 1.0 / curvature          # signed radius
    # Centre of curvature: perpendicular-left at distance |r|
    cx = x - math.sin(hdg) * r
    cy = y + math.cos(hdg) * r
    angle_start = math.atan2(y - cy, x - cx)
    angle_span = length * curvature   # positive=CCW, negative=CW
    angles = np.linspace(angle_start, angle_start + angle_span, n)
    abs_r = abs(r)
    return [(cx + abs_r * math.cos(a), cy + abs_r * math.sin(a)) for a in angles]


def _road_centerline(road: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Return centre-line sample points for a road from its plan_view."""
    points: List[Tuple[float, float]] = []
    for seg in road.get("plan_view", []):
        x = float(seg["x"])
        y = float(seg["y"])
        hdg = float(seg["hdg"])
        length = float(seg["length"])
        geom = seg.get("geometry", "line")
        if geom == "arc":
            pts = _arc_points(x, y, hdg, length, float(seg.get("curvature", 0.0)))
        else:
            pts = _line_points(x, y, hdg, length)
        points.extend(pts)
    return points


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_dsl_topdown(dsl: Dict[str, Any], output_path: str) -> str:
    """Render a Road DSL as a top-down 2D PNG.

    Draws each road's centre-line with a colour-coded band whose visual width
    reflects the sum of left + right lane widths.  Start points are marked
    with circles, end points with squares.  Road IDs are annotated at the
    mid-point of each centre-line.

    Args:
        dsl: Parsed Road DSL dictionary (must contain ``roads``).
        output_path: File path for the output PNG.

    Returns:
        ``output_path`` (for chaining).
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Road DSL – top-down view", fontsize=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    palette = plt.cm.tab10.colors
    legend_patches = []

    for idx, road in enumerate(dsl.get("roads", [])):
        rid = road.get("id", idx)
        color = palette[idx % len(palette)]
        points = _road_centerline(road)
        if len(points) < 2:
            continue
        xs, ys = zip(*points)

        # Estimate total road width for the shaded band
        first_section = (road.get("lane_sections") or [{}])[0]
        right_lanes = first_section.get("right", [])
        left_lanes = first_section.get("left", [])
        road_width_m = sum(float(l.get("width", 3.5)) for l in right_lanes + left_lanes)
        # Map metres to points: 1 pt ≈ 0.35 mm at 150 dpi → rough scale for display
        # Use a fixed minimum so thin roads are still visible.
        band_lw = max(3.0, road_width_m * 1.5)

        # Shaded road band
        ax.plot(xs, ys, color=color, linewidth=band_lw, alpha=0.20,
                solid_capstyle="round", solid_joinstyle="round")
        # Centre-line
        ax.plot(xs, ys, color=color, linewidth=1.5, linestyle="--", alpha=0.85)

        # Start (circle) and end (square) markers
        ax.plot(*points[0], "o", color=color, markersize=7, zorder=5)
        ax.plot(*points[-1], "s", color=color, markersize=7, zorder=5)

        # Road ID annotation at midpoint
        mid = points[len(points) // 2]
        ax.annotate(
            f"R{rid}",
            xy=mid,
            fontsize=8,
            color=color,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
        )

        legend_patches.append(mpatches.Patch(color=color, label=f"Road {rid}"))

    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
                  framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
