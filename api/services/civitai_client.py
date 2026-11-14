#!/usr/bin/env python
"""
Civitai Client

Thin async wrapper around the Civitai REST API (https://civitai.com/api/v1):
search, model/version lookup, enums, and reading a remote safetensors
header via HTTP Range. Returns Civitai's raw JSON; civitai_mapping turns it
into this API's response shapes.
"""

import json
import logging
import struct
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from api.config import get_app_config

logger = logging.getLogger(__name__)

USER_AGENT = "sd-api-backend/2.0"
ENUMS_TTL = 24 * 3600
# Largest safetensors header peek_header() will read (real Flux headers are ~250 KB)
MAX_HEADER_BYTES = 16 * 1024 * 1024
DOWNLOAD_HOSTS = ("civitai.com",)


class CivitaiError(Exception):
    """A failed Civitai request, carrying the HTTP status to return to our client."""

    def __init__(self, status_code: int, message: str, retry_after: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.retry_after = retry_after


def _error_message(response: httpx.Response) -> str:
    """Civitai's own message from an error response, e.g. {"error": ..., "message": ...}."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200] or response.reason_phrase
    if isinstance(body, dict):
        for key in ("message", "error"):
            if isinstance(body.get(key), str):
                return body[key]
    return response.reason_phrase


def raise_for_civitai_status(response: httpx.Response, has_api_key: bool) -> None:
    """Map a Civitai error response to CivitaiError."""
    if response.status_code < 400:
        return
    message = _error_message(response)
    status = response.status_code
    if status == 401:
        hint = "check civitai.api_key" if has_api_key else "set civitai.api_key or CIVITAI_API_KEY"
        raise CivitaiError(401, f"Civitai: {message} ({hint})")
    if status in (403, 404):
        raise CivitaiError(status, f"Civitai: {message}")
    if status == 429:
        raise CivitaiError(429, "Civitai rate limit reached, try again later",
                           retry_after=response.headers.get("Retry-After"))
    raise CivitaiError(502, f"Civitai returned {status}: {message}")


def is_download_url(url: str) -> bool:
    """True if url points at Civitai itself (where the API key may be sent)."""
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in DOWNLOAD_HOSTS


class CivitaiClient:
    """Async Civitai API client with a small in-memory response cache."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30,
        cache_ttl: float = 60,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, Any]] = {}
        # httpx drops the Authorization header when a redirect leaves the
        # origin, so the key isn't sent to Civitai's storage CDN.
        self._http = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            transport=transport,
        )

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    @property
    def http(self) -> httpx.AsyncClient:
        return self._http

    def auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _get(self, path: str, params: Optional[list] = None, ttl: Optional[float] = None) -> Any:
        """GET a Civitai API path, cached for ttl seconds (default cache_ttl)."""
        ttl = self.cache_ttl if ttl is None else ttl
        cache_key = f"{path}?{json.dumps(params or [])}"
        cached = self._cache.get(cache_key)
        if cached and cached[0] > time.monotonic():
            return cached[1]

        try:
            response = await self._http.get(
                f"{self.base_url}{path}", params=params, headers=self.auth_headers()
            )
        except httpx.TimeoutException:
            raise CivitaiError(504, "Civitai did not respond in time")
        except httpx.HTTPError as e:
            raise CivitaiError(502, f"Could not reach Civitai: {e}")
        raise_for_civitai_status(response, self.has_api_key)
        try:
            data = response.json()
        except ValueError:
            raise CivitaiError(502, "Civitai returned an invalid response")

        if ttl > 0:
            self._evict_expired()
            self._cache[cache_key] = (time.monotonic() + ttl, data)
        return data

    def _evict_expired(self) -> None:
        now = time.monotonic()
        for key in [key for key, (expires, _) in self._cache.items() if expires <= now]:
            del self._cache[key]

    async def search(
        self,
        types: List[str],
        base_models: List[str],
        query: Optional[str] = None,
        sort: Optional[str] = None,
        period: Optional[str] = None,
        nsfw: bool = False,
        limit: int = 24,
        cursor: Optional[str] = None,
    ) -> dict:
        """Search models; returns {"items": [...], "metadata": {"nextCursor": ...}}."""
        params = [("types", t) for t in types] + [("baseModels", b) for b in base_models]
        params += [("limit", limit), ("nsfw", "true" if nsfw else "false")]
        for key, value in (("query", query), ("sort", sort), ("period", period), ("cursor", cursor)):
            if value:
                params.append((key, value))
        return await self._get("/models", params)

    async def get_model(self, model_id: int) -> dict:
        return await self._get(f"/models/{model_id}")

    async def get_version(self, version_id: int) -> dict:
        return await self._get(f"/model-versions/{version_id}")

    async def get_version_by_hash(self, sha256: str) -> dict:
        return await self._get(f"/model-versions/by-hash/{sha256}")

    async def get_enums(self) -> dict:
        return await self._get("/enums", ttl=ENUMS_TTL)

    async def known_base_models(self) -> Optional[set]:
        """Civitai's current BaseModel enum, or None if it can't be fetched."""
        try:
            return set((await self.get_enums()).get("BaseModel") or []) or None
        except CivitaiError as e:
            logger.warning(f"Could not fetch Civitai enums, using the base model mapping unchecked: {e}")
            return None

    async def _read_range(self, url: str, start: int, length: int) -> bytes:
        headers = {"Range": f"bytes={start}-{start + length - 1}", **self.auth_headers()}
        try:
            async with self._http.stream("GET", url, headers=headers) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise_for_civitai_status(response, self.has_api_key)
                # A server ignoring Range answers 200 with the whole file from byte 0
                skip = start if response.status_code == 200 else 0
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) >= skip + length:
                        break
                return bytes(data[skip:skip + length])
        except httpx.TimeoutException:
            raise CivitaiError(504, "Civitai download server did not respond in time")
        except httpx.HTTPError as e:
            raise CivitaiError(502, f"Could not reach Civitai download server: {e}")

    async def peek_header(self, download_url: str) -> dict:
        """
        Read the JSON header of a remote safetensors file without
        downloading it: an 8-byte little-endian length, then the header.
        """
        if not is_download_url(download_url):
            raise CivitaiError(400, f"Refusing to read from non-Civitai URL: {download_url}")
        prefix = await self._read_range(download_url, 0, 8)
        if len(prefix) < 8:
            raise CivitaiError(502, "Remote file is too short to be a safetensors file")
        (length,) = struct.unpack("<Q", prefix)
        if not 0 < length <= MAX_HEADER_BYTES:
            raise CivitaiError(502, "Remote file doesn't look like a safetensors file")
        raw = await self._read_range(download_url, 8, length)
        try:
            header = json.loads(raw)
        except ValueError:
            raise CivitaiError(502, "Remote safetensors header is not valid JSON")
        if not isinstance(header, dict):
            raise CivitaiError(502, "Remote safetensors header is not a JSON object")
        return header


# Singleton instance
_client: Optional[CivitaiClient] = None


def get_civitai_client() -> CivitaiClient:
    """Get the singleton Civitai client, configured from parameters.yaml."""
    global _client
    if _client is None:
        config = get_app_config()
        _client = CivitaiClient(
            base_url=config.civitai_base_url,
            api_key=config.civitai_api_key,
            timeout=config.civitai_request_timeout,
            cache_ttl=config.civitai_search_cache_ttl,
        )
    return _client


def set_civitai_client(client: Optional[CivitaiClient]):
    """Set a custom client instance (for testing)."""
    global _client
    _client = client


async def close_civitai_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
