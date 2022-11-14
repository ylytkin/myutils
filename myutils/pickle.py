import pickle as pkl
from pathlib import Path
from typing import Any, Union

__all__ = [
    "load_pickle",
    "save_pickle",
]


def load_pickle(file_path: Union[Path, str]) -> Any:
    """Load a pickle file.

    :param file_path: str or Path
    :return: any
    """

    file_path = Path(file_path)

    with file_path.open("rb") as file:
        return pkl.load(file)


def save_pickle(obj: Any, file_path: Union[Path, str]) -> None:
    """Save an object to a pickle file.

    :param obj: object to pickle (any)
    :param file_path: str or Path
    """

    file_path = Path(file_path)

    with file_path.open("wb") as file:
        pkl.dump(obj, file, protocol=pkl.HIGHEST_PROTOCOL)
