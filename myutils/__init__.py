import subprocess
from typing import Tuple

__all__ = [
    "run_command",
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
