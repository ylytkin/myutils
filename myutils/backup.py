import logging
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence, Tuple, Union

__all__ = [
    "run_backupper",
]

logger = logging.getLogger(__name__)


def _back_up_file(
    file_path: Union[str, Path],
    backup_directory_path: Union[str, Path],
) -> None:
    file_path = Path(file_path).resolve()
    backup_directory_path = Path(backup_directory_path).resolve()

    logger.info(f"back up file {file_path} to directory {backup_directory_path}")

    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_file_name = f"{now}_{file_path.name}"
    backup_file_path = backup_directory_path / backup_file_name

    shutil.copy(file_path, backup_file_path)


def _remove_oldest_backups(
    backup_directory_path: Union[str, Path],
    num_backups: int = 30,
) -> None:
    backup_directory_path = Path(backup_directory_path).resolve()

    backup_file_paths = sorted(
        backup_directory_path.iterdir(),
        key=lambda path: path.name,
        reverse=True,
    )

    for file_path in backup_file_paths[num_backups:]:
        logger.debug(f"remove oldest back up file {file_path}")

        file_path.unlink()


def _run_backupper_step(
    file_directory_pairs: Sequence[Tuple[Union[str, Path], Union[str, Path]]],
) -> None:
    for file_path, backup_directory_path in file_directory_pairs:
        _back_up_file(file_path, backup_directory_path)
        _remove_oldest_backups(backup_directory_path)


def run_backupper(
    file_directory_pairs: Sequence[Tuple[Union[str, Path], Union[str, Path]]],
    step_hours: int = 24,
) -> None:
    """Run backupper. Useful for backing up databases.

    By default backs up every 24 hours. Only keeps 30 latest backups.

    Args:
        file_directory_pairs (Sequence[Tuple[Union[str, Path], Union[str, Path]]]):
            tuples of file path and backup directory path
        step_hours (int, optional): Back up frequence. Defaults to 24.
    """

    logger.info(f"start backupper every {step_hours} hours with files {file_directory_pairs}")

    while True:
        next_run_time = datetime.utcnow() + timedelta(hours=step_hours)

        try:
            _run_backupper_step(file_directory_pairs)
        except Exception as exc:
            logger.exception("caught exception while running back up")

            raise exc

        sleep_seconds = (next_run_time - datetime.utcnow()).total_seconds()
        logger.debug(f"finish back up. sleep for {sleep_seconds} seconds")
        time.sleep(sleep_seconds)


def main() -> None:
    args = sys.argv[1:]

    if len(args) == 0 or len(args) % 2 > 0:
        print("Please provide file-directory pairs")

    file_directory_pairs = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]

    run_backupper(file_directory_pairs)


if __name__ == "__main__":
    main()
