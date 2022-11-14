import random
from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from pyvis.network import Network
from scipy import sparse

import networkx as nx
from myutils.colors import COLORS

__all__ = [
    "DistanceConversionError",
    "graph_info",
    "calculate_distances",
    "get_layout",
    "networkx_graph_to_pyvis_network",
]


class NetworkxToolsBaseException(Exception):
    pass


class DistanceCalculationError(NetworkxToolsBaseException):
    def __init__(self, n_edges: int) -> None:
        msg = f"Did not reach the graph diameter after {n_edges} (number of edges) iterations."

        super().__init__(msg)


class DistanceConversionError(NetworkxToolsBaseException):
    def __init__(self, n_errors: int) -> None:
        msg = (
            f"Found {n_errors} non-integer values while converting matrix values. "
            f"This is not your fault, but currently unresolved. "
            f"See the docstring to `calculate_distances` or set `check_result=False`."
        )

        super().__init__(msg)


# pylint: disable=too-many-locals
def graph_info(
    graph: nx.Graph,
    display_using: Callable[
        [Any],
        None,
    ] = print,
) -> None:
    """Display basic information about a graph.

    Note: the connected component structure is calculated using
    the undirected equivalent of the graph.

    :param graph: networkx Graph
    :param display_using: function to use to display
    """

    graph_type = repr(type(graph))
    info = f"{graph_type}\n"
    info += "-" * len(graph_type) + "\n"

    info += f"Nodes, Edges\t\t{graph.number_of_nodes()}, {graph.number_of_edges()}\n"

    is_directed = graph.is_directed()
    info += f"Directed\t\t{is_directed}\n"

    connected_components = sorted(nx.connected_components(graph.to_undirected()), key=len)
    n_connected_components = len(connected_components)
    is_connected = n_connected_components == 1
    info += f"Connected\t\t{is_connected}\n"

    if not is_connected:
        info += f"Number of components\t{n_connected_components}\n"
        info += f"Max component size\t{len(max(connected_components, key=len))}\n"

    adjacency_matrix = nx.to_scipy_sparse_array(graph)

    if is_directed:
        out_degree = adjacency_matrix.sum(axis=1)
        info += f"Degree range (out)\t{out_degree.min()} to {out_degree.max()}\n"

        in_degree = adjacency_matrix.sum(axis=0)
        info += f"Degree range (in)\t{in_degree.min()} to {in_degree.max()}\n"

    else:
        degree = adjacency_matrix.sum(axis=0)
        info += f"Degree range\t\t{degree.min()} to {degree.max()}\n"

    node_label_types = repr(list(set(map(type, graph.nodes))))
    info += f"Node label types\t{node_label_types}\n"

    for node in graph.nodes:
        node_attributes = repr(list(graph.nodes[node]))
        info += f"Node attributes\t\t{node_attributes}\n"
        break

    for node_1, node_2 in graph.edges:
        edge_attributes = repr(list(graph[node_1][node_2]))
        info += f"Edge attributes\t\t{edge_attributes}"
        break

    display_using(info)


def calculate_distances(graph: nx.Graph) -> NDArray[Any]:
    """Calculate all distances in an unweighted undirected networkx graph
    (uses numpy for speed and scipy sparse matrices for memory efficiency).

    :param graph: networkx Graph
    :return: numpy ndarray with integers
    """

    adjacency_matrix = nx.adjacency_matrix(graph.to_undirected()).astype(bool).astype(int)
    distances_matrix = adjacency_matrix.copy()

    while True:
        current_distance = distances_matrix.max()
        zero_distances = sparse.csr_matrix(
            np.ones(adjacency_matrix.shape)
        ) - distances_matrix.astype(bool).astype(int)
        new_distances = (
            distances_matrix.dot(adjacency_matrix).astype(bool).astype(int).multiply(zero_distances)
        )

        if new_distances.count_nonzero() == 0:
            break

        distances_matrix += new_distances * (current_distance + 1)

    distances: NDArray[Any] = distances_matrix.A.astype(float)
    distances[np.where(distances == 0)] = np.inf

    ids = np.arange(distances.shape[0])
    distances[ids, ids] = 0

    return distances


def get_layout(
    graph: nx.Graph,
    layout: Optional[str] = "fr",
    seed: Optional[int] = None,
) -> Dict[Any, NDArray[Any]]:
    """Get the desired layout for the graph using functions from `igraph`
    (which is faster than the `networkx` implementation and also prettier imo).

    Note: requires `python-igraph` package, which is not trivial to install.

    :param graph: networkx Graph
    :param layout: layout name or None
    :param seed: int or None
    :return: layout (dict of coordinates)
    """

    from igraph import Graph  # pylint: disable=import-error,import-outside-toplevel

    if seed is not None:
        random.seed(seed)

    nodes = list(graph.nodes())
    n_nodes = len(nodes)
    node2id = dict(zip(nodes, range(n_nodes)))

    edges = [(node2id[n1], node2id[n2]) for n1, n2 in graph.edges()]

    igraph_graph = Graph()
    igraph_graph.add_vertices(n_nodes)
    igraph_graph.add_edges(edges)

    pos_array = np.array(igraph_graph.layout(layout=layout).coords)
    pos_array = (
        2 * (pos_array - pos_array.min(axis=0)) / (pos_array.max(axis=0) - pos_array.min(axis=0))
        - 1
    )

    pos = {node: np.array(coords) for node, coords in zip(nodes, pos_array)}

    return pos


# pylint: disable=too-many-locals
def networkx_graph_to_pyvis_network(
    graph: nx.Graph,
    node_label: str = "label",
    node_title: str = "title",
    node_size: str = "size",
    node_color: str = "color",
    edge_weight: str = "weight",
    height: str = "650px",
    width: str = "100%",
    notebook: bool = False,
    heading: str = "",
    gravity: int = -1000,
) -> Network:
    node_labels = nx.get_node_attributes(graph, node_label)
    node_titles = nx.get_node_attributes(graph, node_title)
    node_sizes = nx.get_node_attributes(graph, node_size)
    node_colors = nx.get_node_attributes(graph, node_color)
    edge_widths = nx.get_edge_attributes(graph, edge_weight)

    pyvis_network = Network(
        height=height,
        width=width,
        directed=nx.is_directed(graph),
        notebook=notebook,
        heading=heading,
    )

    for node in graph.nodes:
        label = node_labels.get(node, node)
        title = node_titles.get(node, node)
        size = node_sizes.get(node, 10)
        color = node_colors.get(node, COLORS[0])

        pyvis_network.add_node(node, label=label, title=title, size=float(size), color=color)

    for edge in graph.edges:
        width = edge_widths.get(edge, 1)
        pyvis_network.add_edge(*edge, width=float(width))

    pyvis_network.barnes_hut(gravity=gravity)

    return pyvis_network
