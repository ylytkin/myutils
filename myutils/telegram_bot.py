import logging
import time
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter

__all__ = [
    "TelegramBot",
]

logger = logging.getLogger(__name__)

JsonLike = Union[str, bool, Dict[str, Any], List[Any]]


class TelegramBot:
    SESSION = requests.Session()
    N_RETRIES = 10
    SESSION.mount("https://", HTTPAdapter(max_retries=N_RETRIES))

    class TelegramBotError(Exception):
        pass

    class TelegramBotInteractionError(TelegramBotError):
        pass

    def __init__(self, token: str) -> None:
        self.token: str = token
        self.base_api_url: str = f"https://api.telegram.org/bot{self.token}/"
        self.base_file_url: str = f"https://api.telegram.org/file/bot{self.token}/"

    def _interact(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        files: Union[None, Dict[str, BinaryIO], Dict[str, Tuple[str, str, str]]] = None,
        post: bool = False,
        retry: bool = True,
    ) -> JsonLike:
        n_retries = self.N_RETRIES if retry else 0

        kwargs = {
            "method": "POST" if post else "GET",
            "url": self.base_api_url + method,
            "params": params,
            "files": files,
        }

        response_json = None

        while True:
            response = self.SESSION.request(**kwargs)  # type: ignore

            if (response.status_code == 200) and (
                response.headers.get("content-type") == "application/json"
            ):
                response_json = response.json()
                result: Optional[JsonLike] = response_json.get("result")

                if result is not None:
                    return result

            if n_retries > 0:
                n_retries -= 1
                time.sleep(2)
                continue

            break

        msg = (
            f"Could not get response for method {method}. Args: {kwargs}, status "
            f"code: {response.status_code} ({response.reason}), json: {response_json}"
        )
        raise self.TelegramBotInteractionError(msg)

    def get_me(self) -> JsonLike:
        method = "getMe"

        logger.debug(f"run telegram bot method {method}")

        return self._interact(method)

    def get_updates(
        self,
        offset: Optional[int] = None,
        timeout: int = 10,
    ) -> List[Any]:
        method = "getUpdates"

        logger.debug(f"run telegram bot method {method} with offset={offset}, timeout={timeout}")

        params = {
            "timeout": timeout,
        }

        if offset is not None:
            params["offset"] = offset

        result: List[Any] = self._interact(method, params)  # type: ignore

        return result

    def get_file_download_url(self, file_id: str) -> str:
        method = "getFile"
        params = {"file_id": file_id}

        logger.debug(f"run telegram bot method {method}")

        result: Dict[str, Any] = self._interact(method, params)  # type: ignore
        file_path: str = result["file_path"]

        return self.base_file_url + file_path

    def send_text_message(
        self,
        chat_id: Union[int, str],
        text: str,
        parse_mode: str = "html",
        disable_notification: bool = False,
    ) -> JsonLike:
        method = "sendMessage"

        logger.debug(f"run telegram bot method {method}")

        params = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
        }

        return self._interact(method, params, post=True)

    def delete_message(
        self,
        chat_id: Union[int, str],
        message_id: Union[int, str],
        missing_ok: bool = True,
    ) -> bool:
        method = "deleteMessage"

        logger.debug(f"run telegram bot method {method}")

        params = {"chat_id": chat_id, "message_id": message_id}

        try:
            success: bool = self._interact(method, params, post=True, retry=False)  # type: ignore

            return success

        except self.TelegramBotInteractionError as exc:
            if missing_ok:
                return False

            raise exc

    def send_text_as_file(
        self,
        chat_id: Union[int, str],
        text: str,
        file_name: str,
        caption: Optional[str] = None,
        parse_mode: str = "html",
    ) -> JsonLike:
        method = "sendDocument"

        logger.debug(f"run telegram bot method {method}")

        params = {
            "chat_id": chat_id,
            "parse_mode": parse_mode,
        }

        if caption:
            params["caption"] = caption

        files = {
            "document": (file_name, text, "text/plain"),
        }

        result = self._interact(method, params, files, post=True)

        return result

    def send_file(
        self,
        chat_id: Union[int, str],
        file_path: Path,
        caption: Optional[str] = None,
        parse_mode: str = "html",
    ) -> JsonLike:
        method = "sendDocument"

        logger.debug(f"run telegram bot method {method}")

        params = {"chat_id": chat_id, "parse_mode": parse_mode}

        if caption:
            params["caption"] = caption

        with Path(file_path).open("rb") as file:
            files = {"document": file}
            result = self._interact(method, params, files, post=True)  # type: ignore

        return result

    def send_image(
        self,
        chat_id: Union[int, str],
        image_fpath: Path,
        caption: Optional[str] = None,
        parse_mode: str = "html",
    ) -> JsonLike:
        method = "sendPhoto"

        logger.debug(f"run telegram bot method {method}")

        params = {"chat_id": chat_id, "parse_mode": parse_mode}

        if caption:
            params["caption"] = caption

        with Path(image_fpath).open("rb") as file:
            files = {"photo": file}
            result = self._interact(method, params, files, post=True)  # type: ignore

        return result
