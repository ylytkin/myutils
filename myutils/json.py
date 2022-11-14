import json
from pathlib import Path
from typing import Any, Dict, List, Union

__all__ = [
    "load_json",
    "save_json",
    "load_json_or_create",
]

JsonLike = Union[str, bool, Dict[str, Any], List[Any]]


def load_json(
    file_path: Union[Path, str],
    encoding: str = "utf-8",
) -> JsonLike:
    """Load json file.

    Args:
        file_path (Union[Path, str]): file path
        encoding (str, optional): encoding. Defaults to "utf-8".

    Returns:
        JsonLike: json file data
    """

    file_path = Path(file_path)

    with file_path.open(encoding=encoding) as file:
        json_data: JsonLike = json.load(file)

    return json_data


def save_json(
    obj: JsonLike,
    file_path: Union[Path, str],
    encoding: str = "utf-8",
    indent: int = 4,
) -> None:
    """Save json-like data to file.

    Args:
        obj (JsonLike): json-like data
        file_path (Union[Path, str]): file path
        encoding (str, optional): encoding. Defaults to "utf-8".
        indent (int, optional): indentation in json file
    """

    file_path = Path(file_path)

    with file_path.open("w", encoding=encoding) as file:
        json.dump(obj, file, ensure_ascii=False, indent=indent)


def load_json_or_create(
    file_path: Union[Path, str],
    factory: type,
    encoding: str = "utf-8",
) -> JsonLike:
    """Load json file, if it exists, and create a factory object otherwise.

    Args:
        file_path (Union[Path, str]): file path
        factory (type): object factory
        encoding (str, optional): encoding. Defaults to "utf-8".

    Returns:
        JsonLike: json file data
    """

    file_path = Path(file_path)

    if file_path.exists():
        return load_json(file_path, encoding)

    json_data: JsonLike = factory()

    return json_data
