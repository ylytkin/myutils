import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, Union

import pandas as pd

from myutils import run_command

__all__ = [
    "run_memory_logger",
]

DATE_COL = "date"
COLUMNS_TO_LOG = [DATE_COL, "used", "available"]


def _get_row(values: Sequence[Any], sep: str) -> str:
    return sep.join(map(str, values)) + "\n"


def _get_memory_log_row(with_header: bool, sep: str) -> str:
    cmd = "free -m"

    out, err = run_command(cmd, strip=True)

    if err:
        raise RuntimeError(f"command '{cmd}' resulted in error: {err}")

    header, mem, _ = out.splitlines()  # third is swap

    header_values = re.split(r"\s+", header)
    mem_values = re.split(r"\s+", mem)

    assert mem_values[0] == "Mem:"
    values: pd.Series = pd.Series(mem_values[1:], index=header_values)
    values[DATE_COL] = datetime.utcnow()

    values = values.loc[COLUMNS_TO_LOG]

    memory_log_row = ""

    if with_header:
        memory_log_row += _get_row(values.index, sep=sep)

    memory_log_row += _get_row(values, sep=sep)

    return memory_log_row


def run_memory_logger(
    file_path: Union[str, Path], sep: str = ",", step_seconds: float = 15
) -> None:
    """Run memory logger, which logs RAM state to file.

    Args:
        file_path (Union[str, Path]): path of file to log to
        sep (str, optional): row values separator. Defaults to ','.
        step_seconds (float, optional): step (in seconds) with which to log. Defaults to 15.
    """

    file_path = Path(file_path).resolve()
    print(f"Logging memory state every {step_seconds} seconds to: {file_path}")

    while True:
        memory_log_row = _get_memory_log_row(with_header=not file_path.exists(), sep=sep)

        with file_path.open("a", encoding="utf-8") as file:
            file.write(memory_log_row)

        time.sleep(step_seconds)


def main() -> None:
    args = sys.argv[1:]

    if len(args) != 1:
        print("Please provide a file path")
        return

    file_path = args[0]

    run_memory_logger(file_path)


if __name__ == "__main__":
    main()
