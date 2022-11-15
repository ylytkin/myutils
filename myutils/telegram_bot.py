import logging
import time
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError

__all__ = [
    "TelegramBot",
]

logger = logging.getLogger(__name__)

JsonLike = Union[str, bool, Dict[str, Any], List[Any]]


class TelegramBot:
    SESSION = requests.Session()
    SESSION.mount("https://", HTTPAdapter(max_retries=10))

    def __init__(self, token: str) -> None:
        self.token: str = token
        self.base_api_url: str = f"https://api.telegram.org/bot{self.token}/"
        self.base_file_url: str = f"https://api.telegram.org/file/bot{self.token}/"

    def _interact(
        self,
        request_method: str,
        api_method: str,
        params: Optional[Dict[str, Any]] = None,
        files: Union[None, Dict[str, BinaryIO], Dict[str, Tuple[str, str, str]]] = None,
    ) -> JsonLike:
        url = self.base_api_url + api_method

        logger.debug(f"request {request_method} via {url} with params {params}, files {files}")

        response = self.SESSION.request(
            request_method,
            url,
            params=params,
            files=files,
        )
        try:
            response.raise_for_status()
        except HTTPError as exc:
            if response.status_code == 504 and response.reason == "Gateway Time-out":
                time.sleep(1)

                return self._interact(
                    request_method=request_method,
                    api_method=api_method,
                    params=params,
                    files=files,
                )

            raise exc

        response_json = response.json()
        result: JsonLike = response_json["result"]

        return result

    def get_me(self) -> JsonLike:
        method = "getMe"

        logger.debug(f"get bot info using method {method}")

        return self._interact("GET", api_method=method)

    def get_updates(
        self,
        offset: Optional[int] = None,
        timeout: int = 10,
    ) -> List[Any]:
        method = "getUpdates"

        logger.debug(f"get updates using method {method}")

        params = {
            "timeout": timeout,
        }

        if offset is not None:
            params["offset"] = offset

        result: List[Any] = self._interact("GET", api_method=method, params=params)  # type: ignore

        return result

    def get_file_download_url(self, file_id: str) -> str:
        method = "getFile"
        params = {"file_id": file_id}

        logger.debug(f"get file download url using method {method}")

        result: Dict[str, Any] = self._interact(
            "GET",
            api_method=method,
            params=params,
        )  # type: ignore
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

        logger.debug(f"send text message using method {method}")

        params = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
        }

        return self._interact("POST", api_method=method, params=params)

    def delete_message(
        self,
        chat_id: Union[int, str],
        message_id: Union[int, str],
        missing_ok: bool = True,
    ) -> None:
        method = "deleteMessage"

        logger.debug(f"delete message using method {method} with missing_ok={missing_ok}")

        params = {"chat_id": chat_id, "message_id": message_id}

        try:
            self._interact("POST", api_method=method, params=params)
        except HTTPError as exc:
            if (
                missing_ok
                and exc.response.status_code == 400
                and exc.response.reason == "Bad Request"
            ):
                return

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

        logger.debug(f"send text as file using method {method}")

        params = {
            "chat_id": chat_id,
            "parse_mode": parse_mode,
        }

        if caption:
            params["caption"] = caption

        files = {
            "document": (file_name, text, "text/plain"),
        }

        result = self._interact("POST", api_method=method, params=params, files=files)

        return result

    def send_file(
        self,
        chat_id: Union[int, str],
        file_path: Path,
        caption: Optional[str] = None,
        parse_mode: str = "html",
    ) -> JsonLike:
        method = "sendDocument"

        logger.debug(f"send file using method {method}")

        params = {"chat_id": chat_id, "parse_mode": parse_mode}

        if caption:
            params["caption"] = caption

        with Path(file_path).open("rb") as file:
            files = {"document": file}
            result = self._interact(
                "POST",
                api_method=method,
                params=params,
                files=files,  # type: ignore
            )

        return result

    def send_image(
        self,
        chat_id: Union[int, str],
        image_file_path: Path,
        caption: Optional[str] = None,
        parse_mode: str = "html",
    ) -> JsonLike:
        method = "sendPhoto"

        logger.debug(f"send image using method {method}")

        params = {"chat_id": chat_id, "parse_mode": parse_mode}

        if caption:
            params["caption"] = caption

        with Path(image_file_path).open("rb") as file:
            files = {"photo": file}
            result = self._interact(
                "POST",
                api_method=method,
                params=params,
                files=files,  # type: ignore
            )

        return result
