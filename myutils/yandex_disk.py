import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError

__all__ = [
    "YandexDiskAPI",
]

logger = logging.getLogger(__name__)

JsonLike = Union[str, bool, Dict[str, Any], List[Any]]


class YandexDiskAPI:
    YANDEX_DISK_API_TOKEN_KEY = "YANDEX_DISK_API_TOKEN"
    YANDEX_DISK_API_BASE_URL = "https://cloud-api.yandex.net/v1/disk/"

    SESSION = requests.Session()
    SESSION.mount("https://", HTTPAdapter(max_retries=10))

    def __init__(self, token: Optional[str] = None) -> None:
        self.token = self._get_token(token)
        self.headers = self._get_headers(self.token)

    @classmethod
    def _get_token(cls, token: Optional[str]) -> str:
        if token is None:
            token = os.environ.get(cls.YANDEX_DISK_API_TOKEN_KEY)

        if token is None:
            raise KeyError(
                f"yandex disk api token must be provided via either "
                f"function attribute or environment variable {cls.YANDEX_DISK_API_TOKEN_KEY}"
            )

        return token

    @staticmethod
    def _get_headers(token: str) -> Dict[str, str]:
        return {
            "Authorization": f"OAuth {token}",
            "Content-Type": "application/json",
        }

    def _interact(
        self,
        request_method: str,
        api_method: str,
        params: Dict[str, Any],
    ) -> JsonLike:
        url = self.YANDEX_DISK_API_BASE_URL + api_method

        if not params["path"].startswith("disk:/"):
            raise ValueError("yandex disk path must start with disk:/")

        logger.debug(f"request {request_method} via {url} with params {params}")

        response = self.SESSION.request(request_method, url, params=params, headers=self.headers)
        response.raise_for_status()

        response_json: JsonLike = response.json()

        return response_json

    def get_object_info(self, path: str) -> JsonLike:
        method = "resources"
        params = {"path": path, "limit": 1}

        logger.debug(f"get file info using method {method}")

        return self._interact("GET", api_method=method, params=params)

    def publish_object(self, path: str) -> None:
        method = "resources/publish"
        params = {"path": path}

        logger.debug(f"publish file using method {method}")

        self._interact("PUT", api_method=method, params=params)

    def publish_object_and_get_link(self, path: str) -> str:
        self.publish_object(path)
        object_info: Dict[str, Any] = self.get_object_info(path)  # type: ignore

        object_link: str = object_info["public_url"]

        return object_link

    def create_folder(self, path: str, exist_ok: bool = True) -> None:
        method = "resources"
        params = {"path": path}

        logger.debug(f"create folder using method {method}")

        try:
            self._interact("PUT", api_method=method, params=params)

        except HTTPError as exc:
            if exist_ok and exc.response.status_code == 409 and exc.response.reason == "CONFLICT":
                return

            raise exc

    def get_file_list(self, path: str, files_only: bool = False) -> List[Dict[str, Any]]:
        method = "resources"
        limit = 20
        offset = 0
        total = None

        logger.debug(f"get file list using method {method}")

        file_list: List[Dict[str, Any]] = []

        while total is None or offset < total:
            params = {"path": path, "limit": limit, "offset": offset}

            logger.debug(f"> offset {offset} (total {total})")

            response: Dict[str, Any] = self._interact(  # type: ignore
                "GET",
                api_method=method,
                params=params,
            )

            try:
                result: Dict[str, Any] = response["_embedded"]
            except KeyError as exc:
                raise ValueError(f"path is not a directory: {path}") from exc

            total = result["total"]
            offset += limit

            file_list.extend(result["items"])

        if files_only:
            file_list = [file for file in file_list if file["type"] == "file"]

        return file_list

    def get_file_paths_list(self, path: str) -> List[str]:
        file_list = self.get_file_list(path, files_only=False)

        file_paths_list: List[str] = [file_info["path"] for file_info in file_list]

        return file_paths_list

    def _get_file_download_url(self, file_path: str) -> str:
        method = "resources/download"
        params = {"path": file_path}

        logger.debug(f"get file download url using method {method}")

        response: Dict[str, Any] = self._interact(  # type: ignore
            "GET",
            api_method=method,
            params=params,
        )
        download_url: str = response["href"]

        return download_url

    def download_file(self, disk_file_path: str, local_file_path: Union[str, Path]) -> None:
        url = self._get_file_download_url(disk_file_path)

        logger.debug(f"download file {disk_file_path} to {local_file_path}")

        response = self.SESSION.get(url, headers=self.headers)
        response.raise_for_status()

        local_file_path = Path(local_file_path)

        with local_file_path.open("wb") as file:
            file.write(response.content)

    def _get_file_upload_url(
        self,
        disk_file_path: str,
        overwrite: bool = False,
    ) -> str:
        method = "resources/upload"
        params = {"path": disk_file_path, "overwrite": overwrite}

        logger.debug(f"get file upload url using method {method}")

        response: Dict[str, Any] = self._interact(  # type: ignore
            "GET",
            api_method=method,
            params=params,
        )

        upload_url: str = response["href"]

        return upload_url

    def upload_file(
        self,
        local_file_path: Union[str, Path],
        disk_file_path: str,
        overwrite: bool = False,
    ) -> None:
        url = self._get_file_upload_url(disk_file_path, overwrite=overwrite)

        logger.debug(f"upload file {local_file_path} to {disk_file_path}")

        local_file_path = Path(local_file_path)

        with local_file_path.open("rb") as file:
            response = self.SESSION.put(url, data=file, headers=self.headers)
            response.raise_for_status()

    def delete_object(self, path: str) -> None:
        method = "resources"
        params = {"path": path}

        logger.debug(f"delete file using method {method}")

        self._interact("DELETE", api_method=method, params=params)

    def move_object(
        self,
        disk_source_path: str,
        disk_target_path: str,
        overwrite: bool = False,
    ) -> None:
        method = "resources/move"
        params = {
            "from": disk_source_path,
            "path": disk_target_path,
            "overwrite": overwrite,
        }

        logger.debug(f"move file using method {method}")

        self._interact("POST", api_method=method, params=params)
