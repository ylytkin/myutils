import json
import logging
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import requests
from requests.exceptions import HTTPError

from myutils.utils import create_requests_session

__all__ = [
    "TelegramBot",
]

logger = logging.getLogger(__name__)

JsonLike = Union[str, bool, Dict[str, Any], List[Any]]


class TelegramBot:
    SESSION = create_requests_session()

    def __init__(self, token: str) -> None:
        self.token: str = token
        self.base_api_url: str = f"https://api.telegram.org/bot{self.token}/"
        self.base_file_url: str = f"https://api.telegram.org/file/bot{self.token}/"

    @staticmethod
    def _add_telegram_api_error_info_to_exception_inplace(
        response: requests.Response,
        exception: Exception,
    ) -> None:
        exception_args_list = list(exception.args)
        exception_message = exception_args_list[0]

        error_description = response.json().get("description")
        new_exception_message = f"{exception_message} /// from Telegram API: {error_description}"

        exception_args_list[0] = new_exception_message

        exception.args = tuple(exception_args_list)

    def _interact(
        self,
        request_method: str,
        api_method: str,
        params: Optional[Dict[str, Any]] = None,
        files: Union[None, Dict[str, BinaryIO], Dict[str, Tuple[str, str, str]]] = None,
        timeout: int = 15,  # must be >10, or will conflict with long polling
    ) -> JsonLike:
        url = self.base_api_url + api_method

        logger.debug(
            f"request {request_method} via {url} with params={params}, "
            f"files={files}, timeout={timeout}"
        )

        response = self.SESSION.request(
            request_method,
            url,
            params=params,
            files=files,
            timeout=timeout,
        )

        try:
            response.raise_for_status()

        except HTTPError as exc:
            self._add_telegram_api_error_info_to_exception_inplace(response, exc)
            raise exc

        result: JsonLike = response.json()["result"]

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

        result: List[Any] = self._interact(  # type: ignore
            "GET",
            api_method=method,
            params=params,
        )

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

    def edit_text_message(
        self,
        chat_id: Union[int, str],
        message_id: Union[int, str],
        text: str,
        parse_mode: str = "html",
    ) -> JsonLike:
        method = "editMessageText"

        logger.debug(f"edit text message using method {method}")

        params = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": parse_mode,
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
            telegram_error_message = exc.response.json().get("description", "").lower()

            if missing_ok and exc.response.status_code == 400:
                if "message to delete not found" in telegram_error_message:
                    logger.debug("message was deleted by user")
                    return None

                if "message can't be deleted for everyone" in telegram_error_message:
                    logger.debug("message is too old and cannot be deleted")
                    return None

            raise exc

        return None

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

    @staticmethod
    def _get_max_size_photo_id(
        photos: List[Dict[str, Any]],
    ) -> str:
        def file_size_sorter(photo: Dict[str, Any]) -> int:
            file_size: int = photo["file_size"]

            return file_size

        max_size_photo = max(photos, key=file_size_sorter)
        file_id: str = max_size_photo["file_id"]

        return file_id

    def edit_image(
        self,
        chat_id: Union[int, str],
        message_id: Union[int, str],
        buffer_chat_id: Union[int, str],
        image_file_path: Path,
        caption: Optional[str] = None,
        parse_mode: str = "html",
    ) -> Optional[JsonLike]:
        image_file_path = Path(image_file_path)

        buffer_data: str = json.dumps(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "image_file_path": image_file_path.as_posix(),
                "caption": caption,
                "parse_mode": parse_mode,
            }
        )
        logger.debug(f"create buffer message to upload edited image: {buffer_data}")

        buffer_message: Dict[str, Any] = self.send_image(  # type: ignore
            chat_id=buffer_chat_id,
            image_file_path=image_file_path,
            caption=f"<code>{buffer_data}</code>",
            parse_mode="HTML",
        )
        image_file_id = self._get_max_size_photo_id(buffer_message["photo"])

        method = "editMessageMedia"

        logger.debug(f"edit image using method {method}")

        params = {
            "chat_id": chat_id,
            "message_id": message_id,
            "media": json.dumps(
                {
                    "type": "photo",
                    "media": image_file_id,
                    "caption": caption,
                    "parse_mode": parse_mode,
                }
            ),
        }

        result = None

        try:
            result = self._interact("POST", api_method=method, params=params)
        except HTTPError as exc:
            if exc.response.status_code != 400:
                raise exc

            error_description = exc.response.json().get("description", "").lower()

            if "message is not modified" not in error_description:
                raise exc

            logger.debug("edited image was not modified")

        self.delete_message(chat_id=buffer_chat_id, message_id=buffer_message["message_id"])

        return result

    @staticmethod
    def extract_message_command(message: Dict[str, Any]) -> Optional[str]:
        entities = message.get("entities", [])
        commands = [entity for entity in entities if entity["type"] == "bot_command"]

        if len(commands) == 0:
            return None

        if len(commands) > 1:
            logger.warning(f"multiple commands in a message. extracting only first one. {message}")

        command_item = commands[0]
        offset = command_item["offset"]
        length = command_item["length"]

        if offset != 0:
            return None

        command: str = message["text"][offset:length]

        return command

    @staticmethod
    def extract_chat_and_message_id(message: Dict[str, Any]) -> Tuple[int, int]:
        chat_id: int = message["chat"]["id"]
        message_id: int = message["message_id"]

        return chat_id, message_id
