import pickle as pkl
from pathlib import Path
from typing import Union
from typing import Any

from myutils import get_path

__all__ = [
    'load_pickle',
    'save_pickle',
]


def load_pickle(fpath: Union[Path, str]) -> Any:
    """Load a pickle file.

    :param fpath: str or Path
    :return: any
    """

    fpath = get_path(fpath)

    with fpath.open('rb') as file:
        return pkl.load(file)


def save_pickle(obj: Any, fpath: Union[Path, str]) -> None:
    """Save an object to a pickle file.

    :param obj: object to pickle (any)
    :param fpath: str or Path
    """

    fpath = get_path(fpath)

    with fpath.open('wb') as file:
        pkl.dump(obj, file, protocol=pkl.HIGHEST_PROTOCOL)
