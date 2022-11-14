import logging
import re
from typing import Any

from myutils.telegram_bot import TelegramBot

__all__ = [
    "TelegramLogHandler",
    "TelegramLogFormatter",
]


class TelegramLogFormatter(logging.Formatter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def formatException(self, *args: Any, **kwargs: Any) -> str:
        string = super().formatException(*args, **kwargs)

        return f"```\n{string}\n```"

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


class TelegramLogHandler(logging.Handler):
    def __init__(
        self, token: str, chat_id: int, max_message_length: int = 4000, preview_length: int = 1000
    ) -> None:
        super().__init__()

        self.bot = TelegramBot(token)
        self.chat_id = chat_id

        self.max_message_length = max_message_length
        self.preview_length = preview_length

    def _emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)

        if len(message) < self.max_message_length:
            self.bot.send_text_message(chat_id=self.chat_id, text=message, parse_mode="markdown")
            return

        preview = message[: self.preview_length]

        if len(re.findall("```", preview)) % 2 > 0:
            preview += "\n```"

        self.bot.send_text_message(chat_id=self.chat_id, text=preview, parse_mode="markdown")
        self.bot.send_text_as_file(chat_id=self.chat_id, text=message, file_name="traceback.txt")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._emit(record)

        except Exception:  # pylint: disable=broad-except
            self.handleError(record)
