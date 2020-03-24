from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import networkx as nx

__all__ = [
    'get_layout',
]


def get_layout(g: nx.Graph, layout: Optional[str] = 'fr') -> Dict[Any, np.ndarray]:
    """Get the desired layout for the graph `g` using functions from `igraph` (faster than
    the `networkx` implementation and also prettier imo).

    Note: requires `igraph` package, which is not trivial to install.

    :param g: networkx Graph
    :param layout: layout name or None
    :return: layout (dict of coordinates)
    """

    from igraph import Graph

    nodes = list(g.nodes())
    n_nodes = len(nodes)
    node2id = dict(zip(nodes, range(n_nodes)))

    edges = [(node2id[n1], node2id[n2]) for n1, n2 in g.edges()]

    g = Graph()
    g.add_vertices(n_nodes)
    g.add_edges(edges)

    layout = np.array(g.layout(layout).coords)
    layout = 2 * (layout - layout.min(axis=0)) / (layout.max(axis=0) - layout.min(axis=0)) - 1

    layout = {
        node: np.array(coords)
        for node, coords in zip(nodes, layout)
    }

    return layout
