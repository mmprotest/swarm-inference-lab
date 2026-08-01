"""Small dependency-free JSON HTTP transport used by local engine adapters."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def post_json(endpoint: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        endpoint.rstrip("/") + "/" + path.lstrip("/"),
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from backend: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"backend endpoint unavailable: {exc}") from exc
    result = json.loads(body)
    if not isinstance(result, dict):
        raise RuntimeError("backend returned a non-object JSON response")
    return result


def get_json(endpoint: str, path: str, timeout: float) -> dict[str, Any]:
    request = Request(endpoint.rstrip("/") + "/" + path.lstrip("/"), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"backend endpoint unavailable: {exc}") from exc
    result = json.loads(body)
    if not isinstance(result, dict):
        raise RuntimeError("backend returned a non-object JSON response")
    return result
