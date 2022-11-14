import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter

__all__ = [
    "YandexDiskAPI",
]

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

        response = self.SESSION.request(request_method, url, params=params, headers=self.headers)
        response.raise_for_status()

        response_json: JsonLike = response.json()

        return response_json

    def get_file_info(self, path: str) -> JsonLike:
        method = "resources"
        params = {"path": path}

        return self._interact("GET", api_method=method, params=params)

    def publish_object(self, path: str) -> JsonLike:
        method = "resources/publish"
        params = {"path": path}

        return self._interact("PUT", api_method=method, params=params)

    def publish_object_and_get_link(self, path: str) -> str:
        self.publish_object(path)
        file_info: Dict[str, Any] = self.get_file_info(path)  # type: ignore

        file_link: str = file_info["public_url"]

        return file_link

    def get_file_list(self, path: str, files_only: bool = True) -> List[Dict[str, Any]]:
        method = "resources"
        limit = 20
        offset = 0
        total = None

        file_list: List[Dict[str, Any]] = []

        while total is None or offset < total:
            params = {"path": path, "limit": limit, "offset": offset}

            response: Dict[str, Any] = self._interact(  # type: ignore
                "GET",
                api_method=method,
                params=params,
            )
            result: Dict[str, Any] = response["_embedded"]

            total = result["total"]
            offset += limit

            file_list.extend(result["items"])

        if files_only:
            file_list = [file for file in file_list if file["type"] == "file"]

        return file_list

    def _get_file_download_url(self, file_path: str) -> str:
        method = "resources/download"
        params = {"path": file_path}

        response: Dict[str, Any] = self._interact(  # type: ignore
            "GET",
            api_method=method,
            params=params,
        )
        download_url: str = response["href"]

        return download_url

    def download_file(self, disk_file_path: str, local_file_path: Union[str, Path]) -> None:
        url = self._get_file_download_url(disk_file_path)

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
    ) -> JsonLike:
        url = self._get_file_upload_url(disk_file_path, overwrite=overwrite)

        local_file_path = Path(local_file_path)

        with local_file_path.open("rb") as file:
            response = self.SESSION.put(url, data=file, headers=self.headers)
            response.raise_for_status()

        response_json: JsonLike = response.json()

        return response_json

    def delete_file(self, file_path: str) -> JsonLike:
        method = "resources"
        params = {"path": file_path}

        response: JsonLike = self._interact("DELETE", api_method=method, params=params)

        return response

    def move_file(
        self,
        disk_from_file_path: str,
        disk_to_file_path: str,
        overwrite: bool = False,
    ) -> JsonLike:
        method = "resources/move"
        params = {
            "from": disk_from_file_path,
            "path": disk_to_file_path,
            "overwrite": overwrite,
        }

        response: JsonLike = self._interact("POST", api_method=method, params=params)

        return response
