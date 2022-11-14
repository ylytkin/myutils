import networkx as nx
from myutils.pickle import load_pickle, save_pickle
from tests.common import get_temp_fpath


def test_pickle():
    temp_fpath = get_temp_fpath("pkl")

    obj = nx.Graph()
    obj.add_edge(0, 1)

    save_pickle(obj, temp_fpath)
    loaded_obj = load_pickle(temp_fpath)

    assert isinstance(loaded_obj, nx.Graph)
    assert obj.nodes == loaded_obj.nodes
    assert obj.edges == loaded_obj.edges

    temp_fpath.unlink()
