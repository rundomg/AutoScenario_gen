import os
import math
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import entropy
from evaluation_metrics import compute_connected_angles


def count_lanes_edges(xml_file):
    """Counts the number of lanes, edges, and calculates the total length of lanes in a SUMO network XML."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    lane_count, edge_count, total_length = 0, 0, 0

    for edge in root.findall("edge"):
        if edge.attrib.get("function") != "internal":  # Exclude internal edges
            edge_count += 1
            lanes = edge.findall("lane")
            lane_count += len(lanes)
            lane_lengths = [float(lane.get("length")) for lane in lanes if lane.get("length")]
            if len(lane_lengths) > 0:
                total_length += sum(lane_lengths) / len(lane_lengths)

    return lane_count, edge_count, total_length


def compute_entropy(data_list):
    """Computes entropy for a given list of numerical values."""
    frequency = defaultdict(int)
    for count in data_list:
        frequency[count] += 1
    total_count = sum(frequency.values())

    return -sum((freq / total_count) * math.log(freq / total_count, 2) for freq in frequency.values())


def plot_histograms(edge_numbers, lane_numbers, total_length, save_fig=False, save_fn=None):
    """Plots histograms for edge count, lane count, and total length."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    data = [edge_numbers, lane_numbers, total_length]
    titles = ["Histogram of Edge Numbers", "Histogram of Lane Numbers", "Histogram of Total Length"]
    colors = ["blue", "green", "red"]

    for ax, d, title, color in zip(axs, data, titles, colors):
        ax.hist(d, bins=20, color=color, alpha=0.7, density=True)
        ax.set_title(title)
        ax.set_xlabel(title.split()[-2])  # Extracts 'Edge', 'Lane', or 'Total'
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fn)
    else:
        plt.show()


def plot_scatters(ax, edge_numbers, total_length, angle_list, num_angles_xml):
    """Plots scatter plots for edge count vs total length and edge count vs angles."""
    colors, markers = ["#BEB8DC", "#FFBE7A"], ["o", "*"]
    ax.scatter(edge_numbers, total_length, color=colors[0], marker=markers[0])
    ax.set_xlabel("Number of Edges", fontsize=12)
    ax.set_ylabel("Total Length", color=colors[0], fontsize=12)

    ax2 = ax.twinx()
    cnt = 0
    for net_idx, num_angle in enumerate(num_angles_xml):
        ax2.scatter([edge_numbers[net_idx]] * num_angle, angle_list[cnt: cnt + num_angle], 
                    color=colors[1], marker=markers[1])
        cnt += num_angle
    ax2.set_ylabel("Angles", color=colors[1], fontsize=12)


def list_xml_files(folder_dir):
    """Lists all .net.xml files in the given directory."""
    return [os.path.join(folder_dir, f) for f in os.listdir(folder_dir) if f.endswith("net.xml")]


def compute_mean_std(data_array):
    """Computes mean and standard deviation of a given numerical dataset."""
    return np.mean(data_array), np.std(data_array)


def compute_angle_info(xml_files):
    """Computes angle information from a list of network XML files."""
    angle_list, num_angles_xml = [], []
    for xml_file in xml_files:
        dir_name, net_name = os.path.dirname(xml_file), os.path.basename(xml_file)
        prefix = net_name.split(".")[0]
        node_fn, edge_fn = os.path.join(dir_name, f"{prefix}.nod.xml"), os.path.join(dir_name, f"{prefix}.edg.xml")

        angles = compute_connected_angles(node_fn, edge_fn)
        angle_list.extend(angles.values())
        num_angles_xml.append(len(angles))

    return angle_list, num_angles_xml


def compute_net_states(folder_dir):
    """Computes lane counts, edge counts, and total lengths for all networks in a directory."""
    xml_files = list_xml_files(folder_dir)
    lane_counts, edge_counts, total_lengths = [], [], []

    for xml_file in xml_files:
        lanes, edges, total_length = count_lanes_edges(xml_file)
        lane_counts.append(lanes)
        edge_counts.append(edges)
        total_lengths.append(total_length)

    angle_list, num_angles_xml = compute_angle_info(xml_files)

    return xml_files, lane_counts, edge_counts, total_lengths, angle_list, num_angles_xml


def compute_and_visualize_net_states(folder_dir, save_fig=False, save_fn=None):
    """Computes and visualizes network statistics (histograms and scatter plots)."""
    xml_files, lane_counts, edge_counts, total_lengths, angle_list, num_angles_xml = compute_net_states(folder_dir)

    if save_fig:
        plot_histograms(lane_counts, edge_counts, total_lengths, save_fig=True, save_fn=save_fn)
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_scatters(ax, edge_counts, total_lengths, angle_list, num_angles_xml)
        plt.savefig(os.path.join(os.path.dirname(save_fn), "net_scatter.png"))

    print("Mean ± Std for Lanes, Edges, Length, and Angles:")
    for data in [lane_counts, edge_counts, total_lengths, angle_list]:
        print(compute_mean_std(data))

    return [compute_entropy(data) for data in [lane_counts, edge_counts, total_lengths, angle_list]]


class Evaluator:
    """Evaluates network entropy for lanes, edges, lengths, and angles."""
    
    def __init__(self, result_dir, output_dir):
        self.result_dir = result_dir
        self.output_dir = output_dir

    def compute_states(self, save_fig=False):
        """Computes and saves entropy of lane, edge, and route length distributions."""
        if save_fig:
            os.makedirs(self.output_dir, exist_ok=True)
        
        save_fn = os.path.join(self.output_dir, "net_KL_dist.png")
        lane_entropy, edge_entropy, length_entropy, angle_entropy = compute_and_visualize_net_states(
            self.result_dir, save_fig, save_fn
        )
        print(f"Entropy of Lanes, Edges, Length, and Angles: {lane_entropy}, {edge_entropy}, {length_entropy}, {angle_entropy}")


if __name__ == "__main__":
    categories = ["construction", "general", "intersection"]
    stats = {cat: {"lanes": [], "edges": [], "lengths": []} for cat in categories}
    evaluator = Evaluator("auto_result", "result_analysis")
    evaluator.compute_states()


