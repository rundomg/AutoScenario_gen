"""Quick OpenStreetMap preview tool for CARLA workflows.

This script is designed around the CARLA map workflow documented at:
https://carla.readthedocs.io/en/latest/tuto_G_openstreetmap/

CARLA's official path is:
1. Read `.osm` content.
2. Convert it to `.xodr` with ``carla.Osm2Odr``.
3. Load the resulting OpenDRIVE map in CARLA.

For quick inspection, this tool also renders the road graph in a simple
top-down PNG directly from the `.osm` file, so you can verify the map before
importing it into CARLA.
"""

from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-autoscenario")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EARTH_RADIUS_M = 6_378_137.0
CARLA_DEFAULT_WAY_TYPES = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
]

# Visual grouping for a quick preview. This intentionally keeps non-drivable
# pedestrian/cycle infrastructure hidden by default so the figure is easier to
# compare with the road network CARLA will import.
DEFAULT_PREVIEW_WAY_TYPES = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
]

WAY_STYLES = {
    "motorway": ("#b03a2e", 3.0),
    "motorway_link": ("#cb4335", 2.5),
    "trunk": ("#d35400", 2.8),
    "trunk_link": ("#e67e22", 2.4),
    "primary": ("#ca8a04", 2.6),
    "primary_link": ("#eab308", 2.2),
    "secondary": ("#15803d", 2.2),
    "secondary_link": ("#22c55e", 1.9),
    "tertiary": ("#0369a1", 1.9),
    "tertiary_link": ("#0ea5e9", 1.7),
    "unclassified": ("#475569", 1.6),
    "residential": ("#64748b", 1.4),
    "living_street": ("#7c3aed", 1.4),
    "service": ("#94a3b8", 1.1),
    "road": ("#64748b", 1.2),
}
FALLBACK_STYLE = ("#9ca3af", 1.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an OSM top-down preview and optionally export CARLA OpenDRIVE."
    )
    parser.add_argument(
        "--osm",
        default="data/map.osm",
        help="Path to the input .osm file. Default: data/map.osm",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Default: <osm_basename>_preview.png next to the input file.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Figure title. Default: derived from the input filename.",
    )
    parser.add_argument(
        "--all-highways",
        action="store_true",
        help="Render every way carrying a highway tag instead of the drivable subset.",
    )
    parser.add_argument(
        "--export-xodr",
        default=None,
        help="Optional output .xodr path using CARLA's Osm2Odr conversion.",
    )
    parser.add_argument(
        "--default-lane-width",
        type=float,
        default=4.0,
        help="Lane width passed to carla.Osm2OdrSettings. Default: 4.0",
    )
    parser.add_argument(
        "--generate-traffic-lights",
        action="store_true",
        help="Enable traffic light generation during CARLA OSM->XODR conversion.",
    )
    parser.add_argument(
        "--all-junctions-with-traffic-lights",
        action="store_true",
        help="Force all junctions to generate traffic lights during conversion.",
    )
    return parser.parse_args()


def _derive_output_path(osm_path: str) -> str:
    stem, _ = os.path.splitext(osm_path)
    return f"{stem}_preview.png"


def _load_osm_root(osm_path: str) -> ET.Element:
    tree = ET.parse(osm_path)
    return tree.getroot()


def _tag_map(element: ET.Element) -> Dict[str, str]:
    return {
        tag.attrib["k"]: tag.attrib["v"]
        for tag in element.findall("tag")
        if "k" in tag.attrib and "v" in tag.attrib
    }


def _load_nodes(root: ET.Element) -> Dict[str, Tuple[float, float]]:
    nodes: Dict[str, Tuple[float, float]] = {}
    for node in root.findall("node"):
        node_id = node.attrib.get("id")
        lat = node.attrib.get("lat")
        lon = node.attrib.get("lon")
        if node_id is None or lat is None or lon is None:
            continue
        nodes[node_id] = (float(lat), float(lon))
    return nodes


def _reference_lat_lon(root: ET.Element, nodes: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    bounds = root.find("bounds")
    if bounds is not None:
        min_lat = float(bounds.attrib["minlat"])
        max_lat = float(bounds.attrib["maxlat"])
        min_lon = float(bounds.attrib["minlon"])
        max_lon = float(bounds.attrib["maxlon"])
        return (min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0

    if not nodes:
        raise ValueError("OSM file does not contain any nodes.")

    lats = [lat for lat, _ in nodes.values()]
    lons = [lon for _, lon in nodes.values()]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _latlon_to_local_xy(
    lat: float, lon: float, ref_lat: float, ref_lon: float
) -> Tuple[float, float]:
    ref_lat_rad = math.radians(ref_lat)
    x = EARTH_RADIUS_M * math.radians(lon - ref_lon) * math.cos(ref_lat_rad)
    y = EARTH_RADIUS_M * math.radians(lat - ref_lat)
    return x, y


def _extract_highways(
    root: ET.Element,
    nodes: Dict[str, Tuple[float, float]],
    include_way_types: Optional[Sequence[str]],
    ref_lat: float,
    ref_lon: float,
) -> Tuple[List[Dict[str, object]], Counter]:
    highways: List[Dict[str, object]] = []
    type_counter: Counter = Counter()

    for way in root.findall("way"):
        tags = _tag_map(way)
        highway_type = tags.get("highway")
        if not highway_type:
            continue
        if include_way_types is not None and highway_type not in include_way_types:
            continue

        node_refs = [
            nd.attrib["ref"]
            for nd in way.findall("nd")
            if "ref" in nd.attrib and nd.attrib["ref"] in nodes
        ]
        if len(node_refs) < 2:
            continue

        points = [
            _latlon_to_local_xy(nodes[node_id][0], nodes[node_id][1], ref_lat, ref_lon)
            for node_id in node_refs
        ]
        highways.append(
            {
                "id": way.attrib.get("id", ""),
                "highway": highway_type,
                "name": tags.get("name") or tags.get("name:en") or "",
                "points": points,
            }
        )
        type_counter[highway_type] += 1

    return highways, type_counter


def _style_for_way_type(way_type: str) -> Tuple[str, float]:
    if way_type in WAY_STYLES:
        return WAY_STYLES[way_type]

    base_type = way_type.replace("_link", "")
    return WAY_STYLES.get(base_type, FALLBACK_STYLE)


def _plot_highways(
    highways: Sequence[Dict[str, object]],
    title: str,
    output_path: str,
) -> None:
    if not highways:
        raise ValueError("No matching highway ways were found in the OSM file.")

    grouped: Dict[str, List[Sequence[Tuple[float, float]]]] = defaultdict(list)
    for item in highways:
        grouped[str(item["highway"])].append(item["points"])  # type: ignore[index]

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("white")

    all_xs: List[float] = []
    all_ys: List[float] = []

    for way_type, lines in sorted(grouped.items()):
        color, width = _style_for_way_type(way_type)
        for points in lines:
            xs, ys = zip(*points)
            all_xs.extend(xs)
            all_ys.extend(ys)
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=width,
                alpha=0.95,
                solid_capstyle="round",
                solid_joinstyle="round",
            )

    unique_types = sorted(grouped.keys())
    for way_type in unique_types:
        color, width = _style_for_way_type(way_type)
        ax.plot([], [], color=color, linewidth=max(width, 1.4), label=way_type)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.grid(True, alpha=0.15, linewidth=0.8)

    if all_xs and all_ys:
        margin_x = max((max(all_xs) - min(all_xs)) * 0.05, 5.0)
        margin_y = max((max(all_ys) - min(all_ys)) * 0.05, 5.0)
        ax.set_xlim(min(all_xs) - margin_x, max(all_xs) + margin_x)
        ax.set_ylim(min(all_ys) - margin_y, max(all_ys) + margin_y)

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def _export_xodr_with_carla(
    osm_path: str,
    xodr_path: str,
    default_lane_width: float,
    generate_traffic_lights: bool,
    all_junctions_with_traffic_lights: bool,
) -> None:
    try:
        import carla  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "CARLA Python API is not available in the current environment. "
            "Please activate the CARLA-enabled environment before using --export-xodr."
        ) from exc

    with open(osm_path, "r", encoding="utf-8") as file:
        osm_data = file.read()

    settings = carla.Osm2OdrSettings()
    settings.set_osm_way_types(CARLA_DEFAULT_WAY_TYPES)
    settings.default_lane_width = default_lane_width
    settings.generate_traffic_lights = generate_traffic_lights
    settings.all_junctions_with_traffic_lights = all_junctions_with_traffic_lights

    xodr_data = carla.Osm2Odr.convert(osm_data, settings)

    with open(xodr_path, "w", encoding="utf-8") as file:
        file.write(xodr_data)


def main() -> None:
    args = _parse_args()

    osm_path = os.path.abspath(args.osm)
    output_path = os.path.abspath(args.output or _derive_output_path(osm_path))
    title = args.title or f"OSM Preview: {os.path.basename(osm_path)}"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    root = _load_osm_root(osm_path)
    nodes = _load_nodes(root)
    ref_lat, ref_lon = _reference_lat_lon(root, nodes)
    include_way_types = None if args.all_highways else DEFAULT_PREVIEW_WAY_TYPES
    highways, type_counter = _extract_highways(root, nodes, include_way_types, ref_lat, ref_lon)

    _plot_highways(highways, title, output_path)

    print(f"Rendered preview: {output_path}")
    print(f"Reference origin: lat={ref_lat:.7f}, lon={ref_lon:.7f}")
    print(f"Highway ways rendered: {len(highways)}")
    if type_counter:
        summary = ", ".join(
            f"{way_type}={count}" for way_type, count in sorted(type_counter.items())
        )
        print(f"Way types: {summary}")

    if args.export_xodr:
        xodr_path = os.path.abspath(args.export_xodr)
        os.makedirs(os.path.dirname(xodr_path), exist_ok=True)
        _export_xodr_with_carla(
            osm_path=osm_path,
            xodr_path=xodr_path,
            default_lane_width=args.default_lane_width,
            generate_traffic_lights=args.generate_traffic_lights,
            all_junctions_with_traffic_lights=args.all_junctions_with_traffic_lights,
        )
        print(f"Exported XODR with CARLA: {xodr_path}")


if __name__ == "__main__":
    main()
