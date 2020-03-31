from datetime import datetime
from pathlib import Path
from typing import Union


__all__ = [
    'get_path',
    'log',
]


def get_path(path: Union[str, Path]) -> Path:
    """Convert a string to a Path instance if needed.

    :param path: str or Path
    :return: Path
    """

    if isinstance(path, str):
        path = Path(path)

    return path


def log(*args, to_file: Union[str, Path, None] = None) -> None:
    """Log a message by printing it in stdout along with a timestamp.

    If `to_file` is provided, also appends the message to this file.

    :param args: print args
    :param to_file: file path to log into
    """

    msg = ' '.join(map(lambda x: str(x), args))
    msg = f'[{datetime.now()}] {msg}'

    print(msg)

    if to_file is not None:
        to_file = get_path(to_file)

        with to_file.open('a', encoding='utf-8') as file:
            file.write(msg + '\n')
