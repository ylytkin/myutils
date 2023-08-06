import subprocess
from typing import Tuple

import requests
from requests.adapters import HTTPAdapter, Retry

__all__ = [
    "run_command",
    "escape_html",
    "get_http_adapter",
    "create_requests_session",
]


def run_command(cmd: str, strip: bool = True) -> Tuple[str, str]:
    """Run bash command as subprocess.

    Args:
        cmd (str): command
        strip (bool, optional): whether to strip result strings. Defaults to True.

    Returns:
        Tuple[str, str]: out and err
    """

    with subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        out_bytes, err_bytes = process.communicate()

    out = out_bytes.decode()
    err = err_bytes.decode()

    if strip:
        out = out.strip()
        err = err.strip()

    return out, err


def escape_html(text: str) -> str:
    """Escape HTML characters in string.

    Args:
        text (str): text to process

    Returns:
        str: escaped text
    """

    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def get_http_adapter(
    retries: int = 10,
    backoff_factor: float = 0.5,
) -> HTTPAdapter:
    return HTTPAdapter(
        max_retries=Retry(
            total=retries,
            backoff_factor=backoff_factor,
            allowed_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
            status_forcelist=[500, 502, 503, 504],
        )
    )


def create_requests_session(
    adapter_prefix: str = "https://",
    retries: int = 10,
    backoff_factor: float = 0.5,
) -> requests.Session:
    session = requests.Session()
    session.mount(
        adapter_prefix,
        get_http_adapter(
            retries=retries,
            backoff_factor=backoff_factor,
        ),
    )

    return session
