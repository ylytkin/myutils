import json as jn
from pathlib import Path
from typing import Union

from myutils import get_path

__all__ = [
    'load_json',
    'save_json',
    'load_json_or_create',
]


def load_json(fpath: Union[Path, str], encoding: str = 'utf-8') -> Union[dict, list, str]:
    """Load a json file.

    :param fpath: str or Path
    :param encoding: str
    :return: json file contents
    """

    fpath = get_path(fpath)

    with fpath.open(encoding=encoding) as file:
        return jn.load(file)


def save_json(obj: Union[dict, list, str], fpath: Union[Path, str], encoding: str = 'utf-8') -> None:
    """Save an object to a json file.

    :param obj: dict or list or str
    :param fpath: str or Path
    :param encoding: str
    """

    fpath = get_path(fpath)

    with fpath.open('w', encoding=encoding) as file:
        jn.dump(obj, file, ensure_ascii=False, indent=4)


def load_json_or_create(
        fpath: Union[Path, str],
        factory: type,
        encoding: str = 'utf-8',
) -> Union[dict, list, str]:
    """Load json, if file exists, otherwise create an object
    from the given factory.

    :param fpath: file path
    :param factory: default data type
    :param encoding: encoding
    :return: json file contents
    """

    fpath = Path(fpath)

    if fpath.exists():
        return load_json(fpath, encoding)

    else:
        return factory()
