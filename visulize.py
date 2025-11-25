import rdflib
import networkx as nx
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import os
import random
import numpy as np
from collections import defaultdict
import smg


def shorten_uri(uri):
    """
    Shorten a URI by extracting the fragment or the last part of the path
    """
    if '#' in uri:
        return uri.split('#')[-1]
    elif '/' in uri:
        return uri.split('/')[-1]
    else:
        return uri


def filter_triples(graph, predicate_filter=None, subject_filter=None, object_filter=None):
    """
    Filter RDF triples based on various criteria

    Args:
        graph: RDFlib graph object
        predicate_filter: string to filter predicates (e.g., 'type', 'label')
        subject_filter: string to filter subjects
        object_filter: string to filter objects

    Returns:
        Filtered RDFlib graph
    """
    filtered_graph = rdflib.Graph()

    for s, p, o in graph:
        include = True

        # Filter by predicate
        if predicate_filter and predicate_filter not in str(p):
            include = False

        # Filter by subject
        if subject_filter and subject_filter not in str(s):
            include = False

        # Filter by object
        if object_filter and object_filter not in str(o):
            include = False

        if include:
            filtered_graph.add((s, p, o))

    return filtered_graph


def find_connected_components(graph):
    """
    Find connected components in the RDF graph
    Returns list of subgraphs (connected components)
    """
    nx_graph = nx.DiGraph()

    # Convert to NetworkX graph
    for subject, predicate, obj in graph:
        sub_short = shorten_uri(str(subject))
        pred_short = shorten_uri(str(predicate))
        obj_short = shorten_uri(str(obj))

        nx_graph.add_node(sub_short, type='subject', original=subject)
        nx_graph.add_node(obj_short, type='object', original=obj)
        nx_graph.add_edge(sub_short, obj_short, label=pred_short, original_predicate=predicate)

    # Convert to undirected for connected components
    undirected_graph = nx_graph.to_undirected()
    components = list(nx.connected_components(undirected_graph))

    print(f"Found {len(components)} connected components")

    # Create subgraphs for each component
    component_subgraphs = []
    for i, component_nodes in enumerate(components):
        component_graph = rdflib.Graph()

        # Add all triples that involve nodes in this component
        for subject, predicate, obj in graph:
            sub_short = shorten_uri(str(subject))
            obj_short = shorten_uri(str(obj))

            if sub_short in component_nodes or obj_short in component_nodes:
                component_graph.add((subject, predicate, obj))

        component_subgraphs.append(component_graph)
        print(f"Component {i + 1}: {len(component_graph)} triples, {len(component_nodes)} nodes")

    return component_subgraphs, nx_graph, components


def create_clustered_visualization_high_quality(ttl_file, output_file="rdf_clustered.png", figsize=(20, 16), dpi=400):
    """
    (kept for reference, not used now)
    """
    # Load the RDF graph
    graph = rdflib.Graph()
    graph.parse(ttl_file, format='turtle')
    print(f"Loaded {len(graph)} triples")

    # Find connected components
    component_subgraphs, nx_graph, components = find_connected_components(graph)

    # Create the visualization with ultra high quality
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Use optimized layout for better component separation
    pos = nx.spring_layout(nx_graph, k=3, iterations=300, seed=42)

    # Enhanced color map for better distinction
    colors = plt.cm.tab20(np.linspace(0, 1, len(components)))

    # Calculate optimal sizes based on graph complexity
    total_nodes = len(nx_graph.nodes())
    if total_nodes > 100:
        node_size = 600
        font_size = 7
        edge_font_size = 5
    elif total_nodes > 50:
        node_size = 800
        font_size = 8
        edge_font_size = 6
    else:
        node_size = 1200
        font_size = 10
        edge_font_size = 8

    # Draw each component with distinct colors
    for i, component_nodes in enumerate(components):
        comp_subject_nodes = [node for node in component_nodes
                              if nx_graph.nodes[node].get('type') == 'subject']
        comp_object_nodes = [node for node in component_nodes
                             if nx_graph.nodes[node].get('type') == 'object']

        # Draw subject nodes
        nx.draw_networkx_nodes(nx_graph, pos, nodelist=comp_subject_nodes,
                               node_color=[colors[i]], node_size=node_size,
                               alpha=0.9, edgecolors='black', linewidths=1.0)

        # Draw object nodes with slightly different shape indication
        nx.draw_networkx_nodes(nx_graph, pos, nodelist=comp_object_nodes,
                               node_color=[colors[i]], node_size=node_size,
                               alpha=0.9, edgecolors='black', linewidths=1.0)

    # Draw edges with enhanced styling
    nx.draw_networkx_edges(nx_graph, pos, edge_color='#444444',
                           arrows=True, arrowsize=30, arrowstyle='->',
                           width=2.0, alpha=0.8)

    # Draw labels with improved readability
    nx.draw_networkx_labels(nx_graph, pos, font_size=font_size,
                            font_weight='bold', font_family='DejaVu Sans',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      alpha=0.9, edgecolor='none'))

    # Draw edge labels
    edge_labels = {(u, v): d['label'] for u, v, d in nx_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels,
                                 font_size=edge_font_size, font_family='DejaVu Sans',
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                           alpha=0.9, edgecolor='none'))

    plt.title(f"RDF Triples Visualization - {len(components)} Connected Components",
              fontsize=18, fontweight='bold', pad=25)
    plt.axis('off')
    plt.tight_layout()

    # 现在不再保存 PNG
    # plt.savefig(...)
    plt.show()

    return component_subgraphs, nx_graph


def create_standard_visualization_high_quality(ttl_file, output_file="rdf_standard.png", figsize=(20, 16), dpi=400):
    """
    (kept for reference, not used now)
    """
    # Load the RDF graph
    graph = rdflib.Graph()
    graph.parse(ttl_file, format='turtle')
    print(f"Loaded {len(graph)} triples")

    # Create a NetworkX graph
    nx_graph = nx.DiGraph()

    # Add nodes and edges with shortened labels
    for subject, predicate, obj in graph:
        sub_short = shorten_uri(str(subject))
        pred_short = shorten_uri(str(predicate))
        obj_short = shorten_uri(str(obj))

        nx_graph.add_node(sub_short, type='subject')
        nx_graph.add_node(obj_short, type='object')
        nx_graph.add_edge(sub_short, obj_short, label=pred_short)

    # Create visualization with ultra high quality settings
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Use optimized layout
    try:
        pos = nx.kamada_kawai_layout(nx_graph)
    except:
        pos = nx.spring_layout(nx_graph, k=2, iterations=200, seed=42)

    # Calculate dynamic sizes based on graph complexity
    node_count = len(nx_graph.nodes())

    # Ultra-quality size adjustments
    if node_count > 100:
        node_size = 600
        font_size = 7
        edge_font_size = 5
    elif node_count > 50:
        node_size = 800
        font_size = 8
        edge_font_size = 6
    else:
        node_size = 1200
        font_size = 10
        edge_font_size = 8

    # Draw nodes with enhanced styling
    subject_nodes = [node for node, attr in nx_graph.nodes(data=True) if attr.get('type') == 'subject']
    object_nodes = [node for node, attr in nx_graph.nodes(data=True) if attr.get('type') == 'object']

    nx.draw_networkx_nodes(nx_graph, pos, nodelist=subject_nodes,
                           node_color='#1f77b4', node_size=node_size,
                           alpha=0.95, edgecolors='black', linewidths=1.0)
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=object_nodes,
                           node_color='#2ca02c', node_size=node_size,
                           alpha=0.95, edgecolors='black', linewidths=1.0)

    # Draw edges with enhanced styling
    nx.draw_networkx_edges(nx_graph, pos, edge_color='#444444',
                           arrows=True, arrowsize=30, arrowstyle='->',
                           width=2.0, alpha=0.8)

    # Draw node labels with improved readability
    nx.draw_networkx_labels(nx_graph, pos, font_size=font_size,
                            font_weight='bold', font_family='DejaVu Sans',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      alpha=0.9, edgecolor='none'))

    # Draw edge labels with enhanced positioning
    edge_labels = {(u, v): d['label'] for u, v, d in nx_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels,
                                 font_size=edge_font_size, font_family='DejaVu Sans',
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                           alpha=0.9, edgecolor='none'))

    plt.title("RDF Triples Visualization (Shortened URIs)",
              fontsize=18, fontweight='bold', pad=25)
    plt.axis('off')

    plt.tight_layout()
    # 现在不再保存 PNG
    # plt.savefig(...)

    plt.show()
    return nx_graph, graph


def choose_best_visualization(ttl_file, output_directory, dpi=400):
    """
    Automatically choose the best visualization method based on graph characteristics
    (kept but not used in main; PNG export disabled)
    """
    # Load graph to analyze
    graph = rdflib.Graph()
    graph.parse(ttl_file, format='turtle')

    # Analyze graph structure
    component_subgraphs, _, components = find_connected_components(graph)

    # Decision logic for best visualization
    if len(components) > 1:
        print("🎯 Multiple connected components detected - using CLUSTERED visualization")
        output_file = os.path.join(output_directory, "rdf_BEST_QUALITY_clustered.png")
        return create_clustered_visualization_high_quality(ttl_file, output_file, dpi=dpi)
    else:
        print("🎯 Single connected component detected - using STANDARD visualization")
        output_file = os.path.join(output_directory, "rdf_BEST_QUALITY_standard.png")
        return create_standard_visualization_high_quality(ttl_file, output_file, dpi=dpi)


def export_ultra_quality_vector_formats(ttl_file, output_base_name="rdf_ultra_quality"):
    """
    Export to vector formats for maximum quality - BEST FOR PUBLICATIONS
    (kept but not used now; SVG/PDF/PNG export commented out)
    """
    graph = rdflib.Graph()
    graph.parse(ttl_file, format='turtle')

    nx_graph = nx.DiGraph()
    for subject, predicate, obj in graph:
        sub_short = shorten_uri(str(subject))
        pred_short = shorten_uri(str(predicate))
        obj_short = shorten_uri(str(obj))
        nx_graph.add_node(sub_short, type='subject')
        nx_graph.add_node(obj_short, type='object')
        nx_graph.add_edge(sub_short, obj_short, label=pred_short)

    pos = nx.spring_layout(nx_graph, k=2, iterations=200, seed=42)

    plt.figure(figsize=(20, 16))

    subject_nodes = [node for node, attr in nx_graph.nodes(data=True) if attr.get('type') == 'subject']
    object_nodes = [node for node, attr in nx_graph.nodes(data=True) if attr.get('type') == 'object']

    nx.draw_networkx_nodes(nx_graph, pos, nodelist=subject_nodes,
                           node_color='#1f77b4', node_size=1200, alpha=0.95,
                           edgecolors='black', linewidths=1.0)
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=object_nodes,
                           node_color='#2ca02c', node_size=1200, alpha=0.95,
                           edgecolors='black', linewidths=1.0)
    nx.draw_networkx_edges(nx_graph, pos, edge_color='#444444',
                           arrows=True, arrowsize=25, arrowstyle='->',
                           width=2.0, alpha=0.8)
    nx.draw_networkx_labels(nx_graph, pos, font_size=10, font_weight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      alpha=0.9, edgecolor='none'))

    edge_labels = {(u, v): d['label'] for u, v, d in nx_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels,
                                 font_size=8,
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                           alpha=0.9, edgecolor='none'))

    plt.title("RDF Triples Visualization - Ultra Quality", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    # 所有图像导出全部注释掉，只作为备选参考
    # plt.savefig(f"{output_base_name}.svg", format='svg', bbox_inches='tight')
    # plt.savefig(f"{output_base_name}.pdf", format='pdf', bbox_inches='tight')
    # plt.savefig(f"{output_base_name}_ultra.png", dpi=600, bbox_inches='tight')

    plt.close()


def print_triples_table(graph, max_triples=20):
    """
    Print triples in a table format with shortened URIs
    """
    print("\n" + "=" * 80)
    print("RDF TRIPLES (Shortened URIs)")
    print("=" * 80)
    print(f"{'Subject':<30} {'Predicate':<25} {'Object':<30}")
    print("-" * 80)

    count = 0
    for subject, predicate, obj in graph:
        if count >= max_triples:
            print(f"... and {len(graph) - max_triples} more triples")
            break

        sub_short = shorten_uri(str(subject))
        pred_short = shorten_uri(str(predicate))
        obj_short = shorten_uri(str(obj))

        print(f"{sub_short:<30} {pred_short:<25} {obj_short:<30}")
        count += 1


def create_interactive_visualization(ttl_file, output_html="rdf_interactive.html"):
    """
    Create an interactive visualization using pyvis
    (high-contrast colors, big fonts, white background)
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print("Pyvis not installed. Install with: pip install pyvis")
        return

    # Load the RDF graph
    graph = rdflib.Graph()
    graph.parse(ttl_file, format='turtle')
    print(f"Loaded {len(graph)} triples for interactive HTML")

    # Create pyvis network, white background, black default font
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000"
    )

    # Add nodes and edges
    node_ids = {}
    node_counter = 0

    for subject, predicate, obj in graph:
        sub_short = shorten_uri(str(subject))
        pred_short = shorten_uri(str(predicate))
        obj_short = shorten_uri(str(obj))

        # Add subject node (strong blue)
        if sub_short not in node_ids:
            node_ids[sub_short] = node_counter
            net.add_node(
                node_counter,
                label=sub_short,
                color="#1976d2",       # 深蓝，高对比
                shape="ellipse",
                borderWidth=2,
                font={
                    "size": 26,
                    "color": "#000000",
                    "face": "arial"
                }
            )
            node_counter += 1

        # Add object node (strong orange)
        if obj_short not in node_ids:
            node_ids[obj_short] = node_counter
            net.add_node(
                node_counter,
                label=obj_short,
                color="#ef6c00",       # 橙色，高对比
                shape="box",
                borderWidth=2,
                font={
                    "size": 26,
                    "color": "#000000",
                    "face": "arial"
                }
            )
            node_counter += 1

        # Add edge (dark gray, big label with white outline)
        net.add_edge(
            node_ids[sub_short],
            node_ids[obj_short],
            title=pred_short,
            label=pred_short,
            color="#555555",
            width=2,
            font={
                "size": 22,
                "color": "#000000",
                "strokeWidth": 4,      # 白色描边，类似白底效果
                "strokeColor": "#ffffff",
                "face": "arial",
                "align": "horizontal"
            }
        )

    # Configure physics + global styling (all fonts bigger)
    net.set_options("""
    var options = {
      "nodes": {
        "shadow": false
      },
      "edges": {
        "smooth": {
          "enabled": false
        }
      },
      "physics": {
        "enabled": true,
        "stabilization": {
          "iterations": 150
        }
      }
    }
    """)

    net.write_html(output_html)
    print(f"Interactive visualization saved to: {os.path.abspath(output_html)}")


# Quick filter functions for common use cases
def filter_by_type(graph, type_name=None):
    """Filter triples by type predicate, optionally by specific type"""
    if type_name:
        return filter_triples(graph, predicate_filter='type', object_filter=type_name)
    else:
        return filter_triples(graph, predicate_filter='type')


def filter_by_label(graph):
    """Filter triples by label predicate"""
    return filter_triples(graph, predicate_filter='label')


# MAIN EXECUTION - HTML ONLY
if __name__ == "__main__":
    # Your TTL file path
    ttl_file_path = "./output/vehicle_A_observations_rpi_gpt.ttl"

    # Output directory
    output_directory = "./Visualizations"
    os.makedirs(output_directory, exist_ok=True)

    try:
        print("🔄 Creating INTERACTIVE HTML visualization (only)...")

        # 只生成 HTML，不再生成 PNG / SVG / PDF
        interactive_file = os.path.join(output_directory, "rdf_interactive_bigfont.html")
        create_interactive_visualization(ttl_file_path, interactive_file)

        print("\n✅ DONE.")
        print(f"Open this file in your browser:\n  {interactive_file}")

        while True:
            smg.display_number(112)

    except FileNotFoundError:
        print(f"File {ttl_file_path} not found. Please check the file path.")
    except Exception as e:
        print(f"Error: {e}")
