import logging
import logging.config
from pathlib import Path
from typing import Any, Optional, Union

__all__ = [
    "LogFormatter",
    "configure_logging",
]

logger = logging.getLogger(__name__)


class LogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        message = self.formatMessage(record)

        if record.exc_info:
            record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            if message[-1] != "\n":
                message += "\n"

            message += record.exc_text

        if record.stack_info:
            if message[-1] != "\n":
                message += "\n"

            message += self.formatStack(record.stack_info)

        return message


def _check_file_writeable(file_path: Union[None, str, Path], encoding: str) -> bool:
    if file_path is None:
        return False

    try:
        with open(file_path, "a", encoding=encoding):
            pass
    except FileNotFoundError:
        return False

    return True


# pylint: disable=too-many-locals
def configure_logging(
    *names: Any,
    level: int = logging.DEBUG,
    stdout: bool = False,
    stdout_level: int = logging.INFO,
    file_path: Union[None, str, Path] = None,
    file_level: int = logging.DEBUG,
    file_encoding: str = "utf-8",
    telegram_token: Optional[str] = None,
    telegram_chat_id: Union[None, int, str] = None,
    telegram_level: int = logging.INFO,
    str_format: str = "%(asctime)s %(name)-35s %(levelname)-8s %(message)s",
    msg_format: str = "```\n%(asctime)s\n%(name)s\n%(levelname)s\n\n%(message)s\n```",
    date_format: str = "%Y-%m-%d %H:%M:%S",
    propagate: bool = False,
) -> None:
    formatters = {
        "standard": {
            "class": "myutils.logging.LogFormatter",
            "format": str_format,
            "datefmt": date_format,
        },
        "telegram": {
            "class": "myutils.telegram_logger.TelegramLogFormatter",
            "format": msg_format,
            "datefmt": date_format,
        },
    }

    handlers = {}

    if stdout is True:
        handlers["stdout"] = {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": stdout_level,
        }

    is_file_writeable = _check_file_writeable(file_path, encoding=file_encoding)

    if file_path is not None and is_file_writeable:
        handlers["file"] = {
            "class": "logging.FileHandler",
            "formatter": "standard",
            "filename": file_path,
            "mode": "a",
            "encoding": file_encoding,
            "level": file_level,
        }

    if telegram_token is not None and telegram_chat_id is not None:
        handlers["telegram"] = {
            "class": "myutils.telegram_logger.TelegramLogHandler",
            "formatter": "telegram",
            "token": telegram_token,
            "chat_id": telegram_chat_id,
            "level": telegram_level,
        }

    logger_config_ = {
        "handlers": list(handlers),
        "level": level,
        "propagate": int(propagate),
    }

    loggers = {}

    for name in names:
        loggers[name] = logger_config_.copy()

    logging_config = {
        "version": 1,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": loggers,
    }

    logging.config.dictConfig(logging_config)

    if file_path is not None and not is_file_writeable:
        logger.warning(f"could not configure file handler for path {file_path}")
