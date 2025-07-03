import re
import xml.etree.ElementTree as ET

def parse_edges(xml_file):
    # Load and parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    edge_list = []
    edge_info_tuples = []
    for edge in root.findall("edge"):
        # Check if the edge is not an internal edge, assuming you want to count all edges
        if "function" not in edge.attrib or edge.attrib["function"] != "internal":
            edge_list.append(edge)
            edge_id = edge.get("id")
            from_node = edge.get("from")
            to_node = edge.get("to")
            edge_info_tuples.append((edge_id, from_node, to_node))
    return edge_list, edge_info_tuples


# Function to parse nodes
def parse_nodes(node_file):
    tree = ET.parse(node_file)
    root = tree.getroot()
    nodes = {}
    for node in root.findall("node"):
        node_id = node.get("id")
        x = float(node.get("x"))
        y = float(node.get("y"))
        nodes[node_id] = (x, y)
    return nodes

def read_sumo_file(xml_file_path, skip_section_tag_list=None):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    if skip_section_tag_list:
        for element in root:
            # Skip the configuration section
            if element.tag in skip_section_tag_list:
                root.remove(element)
    return ET.tostring(root, encoding='unicode')

def check_process_finish(results, message):
    stderr_lines = results.stderr.decode('utf-8').splitlines()
    errors = [line for line in stderr_lines if 'warning' not in line.lower()]
    if errors:
        return False
    return True

def strip_out_xml_md(response):
    return response.strip(" ").strip("\n").lstrip("```xml").strip("```").strip("\n")

def extract_text_section(content: str, pattern: str) -> str:
    """Helper function to extract a specific section from text content."""
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else None

def read_file(file_path):
    """
    Read a file and return its content, handling encoding issues.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except UnicodeDecodeError:
        print("UTF-8 decoding failed, attempt to re read using ISO-8859-1 encoding")
        with open(file_path, "r", encoding="ISO-8859-1") as file:
            return file.read().strip()
        
def write_to_file(file_path, content):
    """Writes content to a file with error handling for encoding issues."""
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
    except UnicodeEncodeError:
        print("UTF-8 decoding failed, attempt to re write using ISO-8859-1 encoding")
        with open(file_path, "w", encoding="ISO-8859-1") as file:
            file.write(content)