import logging
from pathlib import Path
from typing import Union, Optional

from telegram_handler import TelegramHandler, HtmlFormatter
from slack_log_handler import SlackLogHandler

__all__ = [
    'get_logger',
]


def get_logger(
    name: Optional[str] = None,
    level: int = logging.DEBUG,
    stdout: bool = False,
    stdout_level: int = logging.INFO,
    file_name: Union[None, str, Path] = None,
    file_level: int = logging.INFO,
    slack_webhook_url: Optional[str] = None,
    slack_level: int = logging.INFO,
    tgbot_token: Optional[str] = None,
    tgbot_chat_id: Optional[int] = None,
    tgbot_level: int = logging.INFO,
    str_format: str = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    msg_format: str = '%(asctime)s\n%(name)s\n%(levelname)s\n\n%(message)s',
    date_format: str = '%Y-%m-%d %H:%M:%S',
):
    str_formatter = logging.Formatter(fmt=str_format, datefmt=date_format)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    current_handlers = [handler.name for handler in logger.handlers]

    if stdout is True:
        stream_handler_name = '_stream'

        if stream_handler_name not in current_handlers:
            stream_handler = logging.StreamHandler()
            stream_handler.set_name(stream_handler_name)
            stream_handler.setLevel(stdout_level)
            stream_handler.setFormatter(str_formatter)
            
            logger.addHandler(stream_handler)
        
    if file_name is not None:
        file_handler_name = '_file'

        if file_handler_name not in current_handlers:
            file_handler = logging.FileHandler(file_name, encoding='utf-8')
            file_handler.set_name(file_handler_name)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(str_formatter)
            
            logger.addHandler(file_handler)
        
    if slack_webhook_url is not None:
        slack_handler_name = '_slack'

        if slack_handler_name not in current_handlers:
            slack_handler = SlackLogHandler(
                slack_webhook_url,
                format=f"```{msg_format}```",
                date_format=date_format,
            )
            slack_handler.name = slack_handler_name
            slack_handler.setLevel(slack_level)

            logger.addHandler(slack_handler)

    if tgbot_token is not None and tgbot_chat_id is not None:
        telegram_handler_name = '_tgbot'

        if telegram_handler_name not in current_handlers:
            telegram_handler = TelegramHandler(tgbot_token, tgbot_chat_id)
            telegram_handler.set_name(telegram_handler_name)
            telegram_handler.setLevel(tgbot_level)
            
            html_formatter = HtmlFormatter(fmt=f"<pre>{msg_format}</pre>", datefmt=date_format)
            telegram_handler.setFormatter(html_formatter)
            
            logger.addHandler(telegram_handler)
        
    return logger
