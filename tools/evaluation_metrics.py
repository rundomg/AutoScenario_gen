import os
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add project root directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.utils import parse_nodes, parse_edges


def count_lanes_edges(xml_file):
    """Counts lanes, edges, and total lane length in a SUMO network XML."""
    edge_list, _ = parse_edges(xml_file)
    lane_count, edge_count, total_length = 0, 0, 0

    for edge in edge_list:
        edge_count += 1
        lane_count += len(edge.findall("lane"))
        total_length += float(edge.attrib["length"])

    return lane_count, edge_count, total_length


def compute_direction_vectors(edges, nodes):
    """Computes direction vectors for each edge."""
    direction_vectors = {}
    for edge_id, from_node, to_node in edges:
        x1, y1 = nodes[from_node]
        x2, y2 = nodes[to_node]
        direction_vectors[edge_id] = (x2 - x1, y2 - y1)
    return direction_vectors


def angle_between_vectors(v1, v2):
    """Computes the angle (in degrees) between two 2D vectors."""
    dot_product = np.dot(v1, v2)
    magnitudes = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = np.clip(dot_product / magnitudes, -1, 1)
    return math.degrees(math.acos(cos_theta))


def compute_connected_angles(node_file, edge_file):
    """Computes angles between connected edges."""
    nodes = parse_nodes(node_file)
    _, edges = parse_edges(edge_file)
    direction_vectors = compute_direction_vectors(edges, nodes)
    connected_angles = {}

    for i, edge1 in enumerate(edges):
        for edge2 in edges[i + 1 :]:
            if set([edge1[1], edge1[2]]) & set(
                [edge2[1], edge2[2]]
            ):  # Check shared nodes
                angle = angle_between_vectors(
                    direction_vectors[edge1[0]], direction_vectors[edge2[0]]
                )
                connected_angles[(edge1[0], edge2[0])] = angle

    return connected_angles


def compute_entropy(data_list):
    """Computes entropy of numerical data based on frequency distribution."""
    frequency = defaultdict(int)
    for count in data_list:
        frequency[count] += 1
    total = sum(frequency.values())

    return -sum(
        (freq / total) * math.log(freq / total, 2) for freq in frequency.values()
    )


def plot_histograms(
    edge_numbers, lane_numbers, total_lengths, save_fig=False, save_fn="edge_dist.png"
):
    """Plots histograms for edge count, lane count, and total length."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    data = [edge_numbers, lane_numbers, total_lengths]
    titles = ["Edge Count", "Lane Count", "Total Length"]
    colors = ["blue", "green", "red"]

    for ax, d, title, color in zip(axs, data, titles, colors):
        ax.hist(d, bins=20, color=color, alpha=0.7)
        ax.set_title(f"Histogram of {title}")
        ax.set_xlabel(title)
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fn)
    else:
        plt.show()


def list_xml_files(folder_dir):
    """Lists all .xml files in a directory."""
    return [
        os.path.join(folder_dir, f)
        for f in os.listdir(folder_dir)
        if f.endswith(".xml")
    ]


def compute_states(folder_dir):
    """Computes lane counts, edge counts, and total lane lengths for all networks in a directory."""
    xml_files = list_xml_files(folder_dir)
    lane_counts, edge_counts, total_lengths = [], [], []

    for xml_file in xml_files:
        lanes, edges, total_length = count_lanes_edges(xml_file)
        lane_counts.append(lanes)
        edge_counts.append(edges)
        total_lengths.append(total_length)

    plot_histograms(lane_counts, edge_counts, total_lengths)

    return compute_entropy(lane_counts), compute_entropy(edge_counts)


def test_single_net(xml_file_path):
    """Tests a single SUMO network XML by counting lanes and edges."""
    lanes, edges, total_length = count_lanes_edges(xml_file_path)
    print(f"Total Edges (excluding internal): {edges}")
    print(f"Total Lanes in the Network: {lanes}")


if __name__ == "__main__":
    data_dir = "auto_result"
    scene_id = "crash_report_interpreter_0000_split"
    node_fn, edge_fn = os.path.join(data_dir, f"{scene_id}.nod.xml"), os.path.join(
        data_dir, f"{scene_id}.edg.xml"
    )
    print(compute_connected_angles(node_fn, edge_fn))
