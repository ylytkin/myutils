from typing import Any, Dict, Optional, Callable

import networkx as nx
import numpy as np
from scipy import sparse

__all__ = [
    'DistanceConversionError',
    'graph_info',
    'calculate_distances',
    'get_layout',
]


class NetworkxToolsBaseException(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class DistanceCalculationError(NetworkxToolsBaseException):
    def __init__(self, n_edges: int) -> None:
        msg = f"Did not reach the graph diameter after {n_edges} (number of edges) iterations."

        super().__init__(msg)


class DistanceConversionError(NetworkxToolsBaseException):
    def __init__(self, n_errors: int) -> None:
        msg = (f"Found {n_errors} non-integer values while converting matrix values. "
               f"This is not your fault, but currently unresolved. "
               f"See the docstring to `calculate_distances` or set `check_result=False`.")

        super().__init__(msg)


def graph_info(g: nx.Graph, display_using: Callable = print) -> None:
    """Display basic information about a graph.

    Note: the connected component structure is calculated using
    the undirected equivalent of the graph.

    :param g: networkx Graph
    :param display_using: function to use to display
    """

    graph_type = repr(type(g))
    info = f'{graph_type}\n'
    info += '-' * len(graph_type) + '\n'

    info += f'Nodes, Edges\t\t{g.number_of_nodes()}, {g.number_of_edges()}\n'

    is_directed = g.is_directed()
    info += f'Directed\t\t{is_directed}\n'

    connected_components = sorted(nx.connected_components(g.to_undirected()), key=len)
    n_connected_components = len(connected_components)
    is_connected = n_connected_components == 1
    info += f'Connected\t\t{is_connected}\n'

    if not is_connected:
        info += f'Number of components\t{n_connected_components}\n'
        info += f'Max component size\t{len(max(connected_components, key=len))}\n'

    adj_mx = nx.to_scipy_sparse_matrix(g)

    if is_directed:
        out_degree = adj_mx.sum(axis=1)
        info += f'Degree range (out)\t{out_degree.min()} to {out_degree.max()}\n'

        in_degree = adj_mx.sum(axis=0)
        info += f'Degree range (in)\t{in_degree.min()} to {in_degree.max()}\n'

    else:
        degree = adj_mx.sum(axis=0)
        info += f'Degree range\t\t{degree.min()} to {degree.max()}\n'

    node_label_types = repr(list(set(map(type, g.nodes))))
    info += f'Node label types\t{node_label_types}\n'

    for u in g.nodes:
        node_attributes = repr(list(g.nodes[u]))
        info += f'Node attributes\t\t{node_attributes}\n'
        break

    for u, v in g.edges:
        edge_attributes = repr(list(g[u][v]))
        info += f'Edge attributes\t\t{edge_attributes}'
        break

    display_using(info)


def calculate_distances(g: nx.Graph) -> np.ndarray:
    """Calculate all distances in an unweighted undirected networkx graph
    (uses numpy for speed and scipy sparse matrices for memory efficiency).

    :param g: networkx Graph
    :return: numpy ndarray with integers
    """

    adj_mx = nx.adjacency_matrix(g.to_undirected()).astype(bool).astype(int)
    distances = adj_mx.copy()

    while True:
        current_distance = distances.max()
        zero_distances = sparse.csr_matrix(np.ones(adj_mx.shape)) - distances.astype(bool).astype(int)
        new_distances = distances.dot(adj_mx).astype(bool).astype(int).multiply(zero_distances)

        if new_distances.count_nonzero() == 0:
            break
        else:
            distances += new_distances * (current_distance + 1)

    distances = distances.A.astype(float)
    distances[np.where(distances == 0)] = np.inf

    ids = np.arange(distances.shape[0])
    distances[ids, ids] = 0

    return distances


def get_layout(g: nx.Graph, layout: Optional[str] = 'fr', seed: Optional[int] = None) -> Dict[Any, np.ndarray]:
    """Get the desired layout for the graph `g` using functions from `igraph`
    (which is faster than the `networkx` implementation and also prettier imo).

    Note: requires `python-igraph` package, which is not trivial to install.

    :param g: networkx Graph
    :param layout: layout name or None
    :param seed: int or None
    :return: layout (dict of coordinates)
    """

    import random
    from igraph import Graph

    if seed is not None:
        random.seed(seed)

    nodes = list(g.nodes())
    n_nodes = len(nodes)
    node2id = dict(zip(nodes, range(n_nodes)))

    edges = [(node2id[n1], node2id[n2]) for n1, n2 in g.edges()]

    g = Graph()
    g.add_vertices(n_nodes)
    g.add_edges(edges)

    layout = np.array(g.layout(layout=layout).coords)
    layout = 2 * (layout - layout.min(axis=0)) / (layout.max(axis=0) - layout.min(axis=0)) - 1

    layout = {
        node: np.array(coords)
        for node, coords in zip(nodes, layout)
    }

    return layout
