import subprocess
from datetime import datetime
from pathlib import Path
from typing import Union, Tuple, Optional


__all__ = [
    'get_path',
    'log',
    'run_command',
    'create_unique_file_path',
]


def get_path(path: Union[str, Path]) -> Path:
    """Convert a string to a Path instance if needed.

    :param path: str or Path
    :return: Path
    """

    return Path(path) if isinstance(path, str) else path


def log(*args) -> None:
    """Log a message by printing it in stdout along with a timestamp.

    :param args: print args
    """

    msg = ' '.join(map(lambda x: str(x), args))
    msg = f'[{datetime.now()}] {msg}'

    print(msg)


def run_command(cmd: str) -> Tuple[str, str]:
    """Run command in a sub-shell.

    :param cmd: command to run
    :return: tuple stdout and stderr
    """
    
    out, err = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE).communicate()
    
    out = out.decode().strip()
    err = err.decode().strip()
    
    return out, err


def create_unique_file_path(file_path: Path) -> Path:
    """Creates a unique file path by adding a number
    at the end of the file name.
    
    Preserves extensions. Does not work if file name
    has multiple or no dot characters.
    """

    if not file_path.exists():
        return file_path

    parent = file_path.parent
    name = file_path.name

    if len([c for c in name if c == '.']) != 1:
        raise ValueError(f'Got multiple dots in file name {repr(name)}')

    name, ext = name.split('.')

    i = 0

    while True:
        i += 1

        file_path = parent / f'{name}-{i}.{ext}'
        
        if not file_path.exists():
            return file_path
