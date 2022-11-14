from pathlib import Path

ROOT_DIRECTORY_PATH = Path(__file__).parent.resolve()

__all__ = [
    "get_temp_fpath",
]


def get_temp_fpath(extension: str) -> Path:
    i = 0
    fpath = ROOT_DIRECTORY_PATH / f".temp_{i}.{extension}"

    while fpath.exists():
        i += 1
        fpath = ROOT_DIRECTORY_PATH / f".temp_{i}.{extension}"

    return fpath
