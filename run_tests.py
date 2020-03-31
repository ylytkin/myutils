from pathlib import Path

import numpy as np
import networkx as nx
from scipy import sparse

from myutils import log
from myutils.json import save_json, load_json, load_json_or_create
from myutils.pickle import save_pickle, load_pickle
from myutils.networkx import graph_info, calculate_distances
from myutils.sparse import normalize_vectors
from myutils.spatial import get_earth_distances
from myutils.text import tokenize_documents

__all__ = [
    'test_log',
    'test_json',
    'test_pickle',
    'test_networkx',
]


root_dir = Path(__file__).absolute().parent


def _get_temp_fpath(extension: str) -> Path:
    i = 0
    fpath = root_dir / f'temp{i}.{extension}'

    while fpath.exists():
        i += 1
        fpath = root_dir / f'temp{i}.{extension}'

    return fpath


def test_log(verbose: bool = True):
    log('Testing', 'log', 1, '...')

    if verbose:
        log('Testing logging to file.')

    log_fpath = _get_temp_fpath('log')

    log_message = 'Test message'

    log(log_message, to_file=log_fpath)

    with log_fpath.open() as file:
        contents = file.readlines()

    assert (len(contents) == 1) and (contents[0].strip().endswith(log_message))

    log_fpath.unlink()

    if verbose:
        log('Done.')


def test_json(verbose: bool = True):
    if verbose is True:
        log('Testing json.')

    temp_fpath = _get_temp_fpath('json')

    obj = {'key1': [1, 2, 3], 'key2': 'string'}

    save_json(obj, temp_fpath)
    load_json(temp_fpath)

    temp_fpath.unlink()

    load_json_or_create(temp_fpath, dict)

    if verbose is True:
        log('Done.')


def test_pickle(verbose: bool = True):
    if verbose is True:
        log('Testing pickle.')

    temp_fpath = _get_temp_fpath('pkl')

    obj = nx.Graph()
    obj.add_edge(0, 1)

    save_pickle(obj, temp_fpath)
    load_pickle(temp_fpath)

    temp_fpath.unlink()

    if verbose is True:
        log('Done.')


def _get_networkx_distance(g: nx.Graph, u, v) -> float:
    try:
        return nx.shortest_path_length(g, u, v)
    except nx.NetworkXNoPath:
        return float('inf')


def test_networkx(n_testing_node_pairs: int = 10 ** 5, verbose: bool = True):
    if verbose is True:
        log('Testing networkx.')

    n_nodes = 2000
    g = nx.erdos_renyi_graph(n_nodes, p=5 / n_nodes)

    graph_info(g)

    if verbose is True:
        log('Calculating distances.')

    distance_mx = calculate_distances(g)

    if verbose is True:
        log(f'Finished calculating distances. Validating results on {n_testing_node_pairs} random node pairs.')

    nodes = list(g.nodes)

    for _ in range(n_testing_node_pairs):
        u, v = np.random.choice(nodes, size=2, replace=False)

        assert _get_networkx_distance(g, u, v) == distance_mx[u, v]

    if verbose is True:
        log('Done.')


def test_sparse(verbose: bool = True):
    if verbose is True:
        log('Testing sparse.')

    mx = np.random.randint(0, 2, size=(15, 20))
    sparse_mx = sparse.csr_matrix(mx)

    for axis in {0, 1}:
        log(f'Testing for axis {axis}.')

        mx_with_normalized_vectors = mx / np.linalg.norm(mx, axis=axis, keepdims=True)
        mx_with_normalized_vectors[np.isnan(mx_with_normalized_vectors)] = 0

        sparse_mx_with_normalized_vectors = normalize_vectors(sparse_mx, axis=axis)

        assert (sparse_mx_with_normalized_vectors == mx_with_normalized_vectors).all()

    if verbose is True:
        log('Done.')


def test_spatial(verbose: bool = True):
    if verbose is True:
        log('Testing spatial.')

    lat_lng1 = np.array([[0, 0]])
    lat_lng2 = np.array([[0, 180], [90, 0]])
    distances_real = np.array([20015.086796020572, 10007.543398010284])

    distances = get_earth_distances(lat_lng1, lat_lng2)

    assert distances.shape == (lat_lng1.shape[0], lat_lng2.shape[0])
    assert (distances.flatten() == distances_real).all()

    if verbose is True:
        log('Done.')


def test_text(verbose: bool = True):
    if verbose is True:
        log('Testing text.')

    documents = ['My cat is a very strange creature.', 'Да ну, это какая-то фигня...']
    tokenized_documents_real = [['cat', 'strange', 'creature'], ['это', 'какой-то', 'фигня']]

    tokenized_documents = tokenize_documents(documents)
    
    assert len(tokenized_documents) == len(tokenized_documents_real)
    assert all(
        tokenized_document == tokenized_document_real
        for tokenized_document, tokenized_document_real in zip(tokenized_documents,
                                                               tokenized_documents_real)
    )
    
    if verbose is True:
        log('Done.')
    

if __name__ == '__main__':
    test_log()
    test_json()
    test_pickle()
    test_networkx()
    test_sparse()
    test_spatial()
    test_text()

    log('Note that the function `get_layout` from `myutils.networkx` was not tested since it requires '
        '`python-igraph`, which is not always trivial to install. Seems to work though.')
