import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, HTTPError
from tqdm import tqdm

from myutils.utils import create_requests_session

__all__ = [
    "NotionAPI",
]

logger = logging.getLogger(__name__)


class NotionAPI:
    SESSION = create_requests_session()

    GET = "GET"
    POST = "POST"
    PATCH = "PATCH"

    REQUEST_METHODS = [GET, POST, PATCH]

    def __init__(
        self,
        token: str,
        version: str = "2022-06-28",
    ) -> None:
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": version,
            "Content-Type": "application/json",
        }

    @staticmethod
    def get_updates_query(last_update_time: datetime) -> Dict[str, Any]:
        last_update_time_string = last_update_time.astimezone(timezone.utc).isoformat()

        query = {
            "or": [
                {
                    "timestamp": date_type,
                    date_type: {
                        "on_or_after": last_update_time_string,
                    },
                }
                for date_type in ["created_time", "last_edited_time"]
            ]
        }

        return query

    # pylint: disable=too-many-branches
    def _request(
        self,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        n_retries: int = 10,
        timeout: int = 30,
    ) -> requests.Response:
        logger.debug(
            f"request {method} {url} with payload={payload}, "
            f"n_retries={n_retries}, timeout={timeout}"
        )

        try:
            if method in [self.POST, self.PATCH]:
                response = self.SESSION.request(
                    method=method,
                    url=url,
                    json=payload,
                    headers=self.headers,
                    timeout=timeout,
                )
            elif method == self.GET:
                response = self.SESSION.request(
                    method=method,
                    url=url,
                    params=payload,
                    headers=self.headers,
                    timeout=timeout,
                )
            else:
                raise ValueError(
                    f"unknown method {method}. available methods {self.REQUEST_METHODS}"
                )

        except (ChunkedEncodingError, ConnectionError) as exc:
            n_retries -= 1

            log_message = (
                f"error while requesting {method} on {url}: {repr(exc)}" f"{n_retries} retries left"
            )

            if n_retries > 5:
                logger.debug(log_message)
            elif n_retries > 0:
                logger.warning(log_message)
            else:
                raise exc

            return self._request(
                method=method,
                url=url,
                payload=payload,
                n_retries=n_retries,
                timeout=timeout,
            )

        try:
            response.raise_for_status()

            return response

        except HTTPError as exc:
            n_retries -= 1

            if response.status_code == 409:
                log_message = (
                    f"conflict error while requesting {method} on {url}. "
                    f"{n_retries} retries left"
                )

                if n_retries > 5:
                    logger.debug(log_message)
                elif n_retries > 0:
                    logger.warning(log_message)
                else:
                    raise exc

            else:
                raise exc

            return self._request(
                method=method,
                url=url,
                payload=payload,
                n_retries=n_retries,
                timeout=timeout,
            )

    def _paginate_through(
        self,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        post: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        page_size = 100

        if limit is not None:
            page_size = min(page_size, limit)

        has_more = True

        if payload is None:
            payload = {}
        else:
            payload = payload.copy()

        payload["page_size"] = page_size

        all_results = []

        while has_more:
            response = self._request(
                method=self.POST if post else self.GET,
                url=url,
                payload=payload,
            )

            response_json = response.json()

            all_results.extend(response_json["results"])

            if limit is not None and len(all_results) >= limit:
                break

            has_more = response_json["has_more"]
            start_cursor = response_json["next_cursor"]

            payload["start_cursor"] = start_cursor

        return all_results

    def list_connected_databases(self) -> List[Dict[str, str]]:
        logger.debug("list connected databases")

        url = "https://api.notion.com/v1/search"

        payload = {
            "filter": {
                "value": "database",
                "property": "object",
            }
        }

        results = self._paginate_through(url=url, payload=payload)

        connected_databases = [
            {
                "id": result["id"],
                "title": "".join(item["plain_text"] for item in result["title"]),
            }
            for result in results
        ]

        return connected_databases

    def fetch_database_pages(
        self,
        database_id: str,
        query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        logger.debug(
            f"fetch pages from database {database_id} with limit {limit} and query {str(query)}"
        )

        url = f"https://api.notion.com/v1/databases/{database_id}/query"

        payload = {}

        if query is not None:
            payload["filter"] = query

        return self._paginate_through(
            url=url,
            payload=payload,
            limit=limit,
        )

    # pylint: disable=too-many-return-statements,too-many-branches
    def extract_value_from_notion_property_value(
        self,
        property_value: Dict[str, Any],
        sep: str = ";",
    ) -> Any:
        value_type = property_value.get("type")

        if value_type is None:
            raise ValueError(f"could not parse property value {property_value}")

        value_contents = property_value[value_type]

        if value_contents is None:
            return None

        if value_type in {"title", "rich_text"}:  # list
            if len(value_contents) == 0:
                return None
            return "".join(item["plain_text"] for item in value_contents)

        if value_type in {"number", "boolean"}:
            return value_contents

        if value_type in {"select", "status"}:
            return value_contents["name"]

        if value_type == "multi_select":  # list
            if len(value_contents) == 0:
                return None
            return sep.join(item["name"] for item in value_contents)

        if value_type == "formula":
            return self.extract_value_from_notion_property_value(value_contents)

        if value_type in {"date", "created_time", "last_edited_time"}:
            value = value_contents

            if isinstance(value, dict):
                value = value["start"]

            if value is not None:
                if value.endswith("Z"):
                    value = value[:-1] + "+00:00"

                value = datetime.fromisoformat(value)

            return value

        if value_type == "checkbox":
            return value_contents

        if value_type == "relation":  # list
            if len(value_contents) == 0:
                return None
            return sep.join(item["id"] for item in value_contents)

        if value_type == "rollup":
            if value_contents["type"] == "array":  # list
                items = value_contents["array"]

                if len(items) == 0:
                    return None

                return sep.join(
                    str(self.extract_value_from_notion_property_value(item)) for item in items
                )

            return self.extract_value_from_notion_property_value(value_contents)

        if value_type == "url":
            return value_contents

        raise ValueError(f"unknown property type for property value {property_value}")

    def convert_notion_properties_to_plain_dict(
        self,
        properties: Dict[str, Any],
    ) -> Dict[str, Any]:
        plain_dict = {
            key: self.extract_value_from_notion_property_value(property_value)
            for key, property_value in properties.items()
        }

        plain_dict = {key: value for key, value in plain_dict.items() if value is not None}

        return plain_dict

    @staticmethod
    def make_notion_property_value(
        property_name: str,
        value: Any,
        schema: Dict[str, str],
        sep: str = ";",
    ) -> Any:
        value_type = schema[property_name]

        if value_type in {"title", "rich_text"}:
            notion_value: Any = [
                {
                    "text": {"content": value},
                },
            ]

        elif value_type == "number":
            notion_value = value

        elif value_type in {"select", "status"}:
            notion_value = {"name": value}

        elif value_type == "date":
            notion_value = {"start": value.isoformat()}

        elif value_type in {"formula", "rollup", "created_time", "last_edited_time"}:
            return None

        elif value_type == "relation":
            notion_value = [{"id": split} for split in value.split(sep)]

        elif value_type == "multi_select":
            notion_value = [{"name": split} for split in value.split(sep)]

        elif value_type == "checkbox":
            notion_value = value

        elif value_type == "url":
            notion_value = value

        else:
            raise ValueError(f"unknown property type {value_type}")

        property_value = {value_type: notion_value}

        return property_value

    def convert_plain_dict_to_notion_properties(
        self,
        plain_dict: Dict[str, Any],
        schema: Dict[str, str],
    ) -> Dict[str, Any]:
        properties = {
            key: self.make_notion_property_value(
                key,
                value,
                schema=schema,
            )
            for key, value in plain_dict.items()
        }

        properties = {key: value for key, value in properties.items() if value is not None}

        return properties

    def update_formula_property(
        self,
        database_id: str,
        property_name: str,
        formula: str,
    ) -> None:
        url = f"https://api.notion.com/v1/databases/{database_id}"

        data = {
            "properties": {
                property_name: {
                    "formula": {
                        "expression": formula,
                    },
                }
            }
        }

        self._request(method=self.PATCH, url=url, payload=data)

    def fetch_database_as_pandas(
        self,
        database_id: str,
        query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        pages = self.fetch_database_pages(database_id, query=query, limit=limit)

        logger.debug("convert fetched pages to pandas")

        dataframe = pd.DataFrame(
            [self.convert_notion_properties_to_plain_dict(page["properties"]) for page in pages]
        )

        logger.debug(f"got dataframe of shape {dataframe.shape}")

        return dataframe

    def get_database_metadata(
        self,
        database_id: str,
    ) -> Dict[str, Any]:
        logger.debug(f"fetch metadata of database {database_id}")

        url = f"https://api.notion.com/v1/databases/{database_id}"

        response = self._request(method=self.GET, url=url)

        database_metadata: Dict[str, Any] = response.json()

        return database_metadata

    def get_database_schema(
        self,
        database_id: str,
    ) -> Dict[str, str]:
        logger.debug(f"fetch schema of database {database_id}")

        database_metadata = self.get_database_metadata(database_id)

        schema = {
            prop_name: prop["type"] for prop_name, prop in database_metadata["properties"].items()
        }

        return schema

    def create_page(
        self,
        database_id: str,
        properties: Dict[str, Any],
        schema: Optional[Dict[str, str]] = None,
    ) -> None:
        logger.debug(f"create page in database {database_id} with properties {str(properties)}")

        if schema is None:
            schema = self.get_database_schema(database_id)

        url = "https://api.notion.com/v1/pages"

        data = {
            "parent": {"database_id": database_id},
            "properties": self.convert_plain_dict_to_notion_properties(properties, schema=schema),
        }

        self._request(method=self.POST, url=url, payload=data)

    def create_pages_from_pandas(
        self,
        database_id: str,
        dataframe: pd.DataFrame,
        progress_bar: bool = False,
    ) -> None:
        logger.debug(
            f"create pages in database {database_id} from dataframe of shape {dataframe.shape}"
        )

        schema = self.get_database_schema(database_id)

        for _, row in tqdm(
            dataframe.iterrows(),
            disable=not progress_bar,
            total=dataframe.shape[0],
        ):
            self.create_page(database_id, properties=row.dropna().to_dict(), schema=schema)

    def update_page(
        self,
        page_id: str,
        database_id: str,
        properties: Dict[str, Any],
        schema: Optional[Dict[str, str]] = None,
    ) -> None:
        logger.debug(
            f"update page {page_id} from database {database_id} with properties {str(properties)}"
        )

        if schema is None:
            schema = self.get_database_schema(database_id)

        url = f"https://api.notion.com/v1/pages/{page_id}"

        data = {
            "properties": self.convert_plain_dict_to_notion_properties(properties, schema=schema),
        }

        self._request(method=self.PATCH, url=url, payload=data)

    def archive_page(
        self,
        page_id: str,
        database_id: str,
    ) -> None:
        logger.debug(f"archive page {page_id} from database {database_id}")

        url = f"https://api.notion.com/v1/pages/{page_id}"

        data = {
            "archived": True,
        }

        response = requests.patch(url, json=data, headers=self.headers)
        response.raise_for_status()

    def unarchive_page(
        self,
        page_id: str,
        database_id: str,
    ) -> None:
        logger.debug(f"unarchive page {page_id} from database {database_id}")

        url = f"https://api.notion.com/v1/pages/{page_id}"

        data = {
            "archived": False,
        }

        response = requests.patch(url, json=data, headers=self.headers)
        response.raise_for_status()

    def fetch_page_contents(
        self,
        page_id: str,
        recursive: bool = True,
        ignore_not_found: bool = False,
    ) -> List[Dict[str, Any]]:
        logger.debug(f"fetch contents of page {page_id}")

        url = f"https://api.notion.com/v1/blocks/{page_id}/children"

        contents = []

        try:
            results = self._paginate_through(url, post=False)

        except HTTPError as exc:
            if exc.response.status_code == 404 and ignore_not_found:
                results = []

            else:
                raise exc

        for block in results:
            contents.append(block)

            if recursive and block["has_children"]:
                contents.extend(
                    self.fetch_page_contents(
                        block["id"],
                        recursive=recursive,
                        ignore_not_found=ignore_not_found,
                    )
                )

        return contents
