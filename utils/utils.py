from datetime import datetime

__all__ = [
    'log',
]


def log(*args) -> None:
    """Equivalent to `print`, but with a time stamp.
    """

    msg = ' '.join(map(lambda x: str(x), args))
    print(f'[{datetime.now()}] {msg}')
