import numpy as np

import networkx as nx
from myutils.networkx import calculate_distances, graph_info

N_NODES = 2000
N_TESTING_NODE_PAIRS = 10**5


def _get_networkx_distance(g: nx.Graph, u, v) -> float:
    try:
        return nx.shortest_path_length(g, u, v)
    except nx.NetworkXNoPath:
        return float("inf")


def test_networkx():
    g = nx.erdos_renyi_graph(N_NODES, p=5 / N_NODES)

    graph_info(g)

    distance_mx = calculate_distances(g)

    nodes = list(g.nodes)

    for _ in range(N_TESTING_NODE_PAIRS):
        u, v = np.random.choice(nodes, size=2, replace=False)

        assert _get_networkx_distance(g, u, v) == distance_mx[u, v]
