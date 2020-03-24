import json as jn
from pathlib import Path
from typing import Union

__all__ = [
    'load_json',
    'save_json',
]


def _get_path(path: Union[str, Path]) -> Path:
    """Convert a string to a Path instance if needed.

    :param path: str or Path
    :return: Path
    """

    if isinstance(path, str):
        path = Path(path)

    return path


def load_json(fpath: Union[Path, str], encoding: str = 'utf-8'):
    """Load a json file.

    :param fpath: str or Path
    :param encoding: str
    :return: dict or list or str
    """

    fpath = _get_path(fpath)

    with fpath.open(encoding=encoding) as file:
        return jn.load(file)


def save_json(json, fpath: Union[Path, str], encoding: str = 'utf-8') -> None:
    """Save json.

    :param json: dict or list or str
    :param fpath: str or Path
    :param encoding: str
    :return:
    """

    fpath = _get_path(fpath)

    with fpath.open('w', encoding=encoding) as file:
        jn.dump(json, file, ensure_ascii=False, indent=4)
