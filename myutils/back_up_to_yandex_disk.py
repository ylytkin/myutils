import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence, Tuple, Union

from myutils.logging import configure_logging
from myutils.yandex_disk import YandexDiskAPI

__all__ = [
    "run_backupper",
]

logger = logging.getLogger(__name__)

YANDEX_DISK_API = YandexDiskAPI()

BASE_DIRECTORY_PATH = "disk:/Apps/Backups"
FALLBACK_DIRECTORY_PATH = "~/.backups"


def _back_up_file(
    file_path: Union[str, Path],
    app_name: str,
) -> None:
    file_path = Path(file_path).resolve()

    logger.info(f"back up file {file_path} from app {app_name}")

    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    back_up_file_name = f"{now}_{file_path.name}"

    try:
        app_back_ups_directory_path = os.path.join(BASE_DIRECTORY_PATH, app_name)
        YANDEX_DISK_API.create_folder(app_back_ups_directory_path, exist_ok=True)

        back_up_file_path = os.path.join(app_back_ups_directory_path, back_up_file_name)
        YANDEX_DISK_API.upload_file(file_path, back_up_file_path, overwrite=False)

    except Exception:  # pylint: disable=broad-except
        logger.exception("could not upload to yandex disk")

        local_directory_path = Path(FALLBACK_DIRECTORY_PATH) / app_name
        local_directory_path.mkdir(exist_ok=True, parents=True)

        local_back_up_file_path = local_directory_path / back_up_file_name

        logger.warning(f"falling back to local back up to {local_back_up_file_path}")

        shutil.copy(file_path, local_back_up_file_path)


def _run_backupper_step(file_app_pairs: Sequence[Tuple[Union[str, Path], str]]) -> None:
    for file_path, app_name in file_app_pairs:
        _back_up_file(file_path, app_name=app_name)


def run_backupper(
    *file_app_pairs: Tuple[Union[str, Path], str],
    step_hours: int = 24,
) -> None:
    """Run backupper. Useful for backing up databases.

    By default backs up every 24 hours.

    Args:
        *file_app_pairs (Tuple[Union[str, Path], str]):
            tuples of file path and app name
        step_hours (int, optional): Back up frequence. Defaults to 24.
    """

    logger.info(f"start backupper every {step_hours} hours with files {file_app_pairs}")

    while True:
        next_run_time = datetime.utcnow() + timedelta(hours=step_hours)

        try:
            _run_backupper_step(file_app_pairs)

        except Exception as exc:
            logger.exception("caught exception while running back up")

            raise exc

        sleep_seconds = (next_run_time - datetime.utcnow()).total_seconds()
        logger.debug(f"finish back up. sleep for {sleep_seconds} seconds")
        time.sleep(sleep_seconds)


def main() -> None:
    configure_logging(
        "__main__",
        "myutils.yandex_disk",
        stdout=True,
        stdout_level=logging.DEBUG,
        telegram_token=os.environ.get("TELEGRAM_LOGGER_TOKEN"),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID"),
        telegram_level=logging.WARNING,
    )

    args = sys.argv[1:]

    if len(args) == 0 or len(args) % 2 > 0:
        print("Please provide file-directory pairs")

    file_app_pairs = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]

    run_backupper(*file_app_pairs)


if __name__ == "__main__":
    main()
