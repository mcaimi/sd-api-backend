"""Civitai client HTTP behaviour, with a mocked transport."""

import asyncio
import json
import struct

import httpx
import pytest

from api.services.civitai_client import CivitaiClient, CivitaiError

BASE = "https://civitai.com/api/v1"


def client_for(handler, api_key="secret", cache_ttl=60):
    return CivitaiClient(BASE, api_key=api_key, cache_ttl=cache_ttl, transport=httpx.MockTransport(handler))


def run(coro):
    return asyncio.run(coro)


def test_search_sends_repeated_filters_and_auth():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"items": [], "metadata": {"nextCursor": "abc"}})

    async def main():
        client = client_for(handler)
        data = await client.search(types=["LORA", "LoCon"], base_models=["SDXL 1.0", "Pony"],
                                   query="detail", sort="Newest", nsfw=False, limit=5, cursor="c1")
        await client.aclose()
        return data

    data = run(main())
    params = seen[0].url.params
    assert params.get_list("types") == ["LORA", "LoCon"]
    assert params.get_list("baseModels") == ["SDXL 1.0", "Pony"]
    assert (params["query"], params["sort"], params["nsfw"], params["limit"], params["cursor"]) == \
        ("detail", "Newest", "false", "5", "c1")
    assert "period" not in params
    assert seen[0].headers["Authorization"] == "Bearer secret"
    assert data["metadata"]["nextCursor"] == "abc"


def test_no_auth_header_without_key():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={})

    async def main():
        client = client_for(handler, api_key=None)
        await client.get_enums()
        await client.aclose()

    run(main())
    assert "Authorization" not in seen[0].headers


def test_responses_are_cached():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"id": 1})

    async def main():
        client = client_for(handler)
        await client.get_model(1)
        await client.get_model(1)
        await client.get_model(2)
        await client.aclose()

    run(main())
    assert len(calls) == 2


def test_cache_can_be_disabled():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"id": 1})

    async def main():
        client = client_for(handler, cache_ttl=0)
        await client.get_model(1)
        await client.get_model(1)
        await client.aclose()

    run(main())
    assert len(calls) == 2


@pytest.mark.parametrize("status,body,expected_status,expected_text", [
    (401, {"error": "Unauthorized", "message": "The creator of this asset requires you to be logged in to download it"},
     401, "requires you to be logged in"),
    (403, {"error": "Early Access", "message": "This asset is in Early Access."}, 403, "Early Access"),
    (404, {"error": "No model with id 1"}, 404, "No model with id 1"),
    (500, {"error": "boom"}, 502, "Civitai returned 500"),
])
def test_errors_are_mapped(status, body, expected_status, expected_text):
    async def main():
        client = client_for(lambda request: httpx.Response(status, json=body))
        try:
            await client.get_model(1)
        finally:
            await client.aclose()

    with pytest.raises(CivitaiError) as exc:
        run(main())
    assert exc.value.status_code == expected_status
    assert expected_text in exc.value.message


def test_401_hint_depends_on_key():
    def handler(request):
        return httpx.Response(401, json={"message": "nope"})

    async def main(api_key):
        client = client_for(handler, api_key=api_key)
        try:
            await client.get_model(1)
        except CivitaiError as e:
            return e.message
        finally:
            await client.aclose()

    assert "set civitai.api_key" in run(main(None))
    assert "check civitai.api_key" in run(main("secret"))


def test_rate_limit_passes_retry_after():
    async def main():
        client = client_for(lambda request: httpx.Response(429, headers={"Retry-After": "30"}, text="slow down"))
        try:
            await client.get_model(1)
        finally:
            await client.aclose()

    with pytest.raises(CivitaiError) as exc:
        run(main())
    assert (exc.value.status_code, exc.value.retry_after) == (429, "30")


def test_timeout_maps_to_504():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    async def main():
        client = client_for(handler)
        try:
            await client.get_model(1)
        finally:
            await client.aclose()

    with pytest.raises(CivitaiError) as exc:
        run(main())
    assert exc.value.status_code == 504


def safetensors_bytes(header: dict) -> bytes:
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"\0" * 64


def range_handler(payload: bytes, seen: list, honour_range=True):
    """Civitai download URL redirecting to a CDN host that serves byte ranges."""
    def handler(request):
        seen.append(request)
        if request.url.host == "civitai.com":
            return httpx.Response(307, headers={"Location": "https://cdn.example.r2.dev/file.safetensors?sig=1"})
        if not honour_range:
            return httpx.Response(200, content=payload)
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        return httpx.Response(206, content=payload[start:end + 1])
    return handler


@pytest.mark.parametrize("honour_range", [True, False])
def test_peek_header_reads_only_the_header(honour_range):
    header = {"__metadata__": {"format": "pt"}, "double_blocks.0.img_attn.norm.key_norm.scale": {"dtype": "BF16"}}
    seen = []

    async def main():
        client = client_for(range_handler(safetensors_bytes(header), seen, honour_range))
        try:
            return await client.peek_header("https://civitai.com/api/download/models/1?type=Model")
        finally:
            await client.aclose()

    assert run(main()) == header


def test_api_key_is_not_sent_to_the_cdn():
    seen = []

    async def main():
        client = client_for(range_handler(safetensors_bytes({"a": {}}), seen))
        try:
            await client.peek_header("https://civitai.com/api/download/models/1")
        finally:
            await client.aclose()

    run(main())
    civitai = [r for r in seen if r.url.host == "civitai.com"]
    cdn = [r for r in seen if r.url.host != "civitai.com"]
    assert civitai and cdn
    assert all(r.headers.get("Authorization") == "Bearer secret" for r in civitai)
    assert all("Authorization" not in r.headers for r in cdn)


def test_peek_header_refuses_other_hosts():
    async def main():
        client = client_for(lambda request: httpx.Response(200))
        try:
            await client.peek_header("https://evil.example/file.safetensors")
        finally:
            await client.aclose()

    with pytest.raises(CivitaiError) as exc:
        run(main())
    assert exc.value.status_code == 400


def test_peek_header_rejects_non_safetensors():
    seen = []

    async def main():
        client = client_for(range_handler(b"<!DOCTYPE html><html>login</html>", seen))
        try:
            await client.peek_header("https://civitai.com/api/download/models/1")
        finally:
            await client.aclose()

    with pytest.raises(CivitaiError) as exc:
        run(main())
    assert exc.value.status_code == 502


def test_peek_header_surfaces_auth_errors():
    def handler(request):
        return httpx.Response(401, json={"error": "Unauthorized", "message": "log in first"})

    async def main():
        client = client_for(handler)
        try:
            await client.peek_header("https://civitai.com/api/download/models/1")
        finally:
            await client.aclose()

    with pytest.raises(CivitaiError) as exc:
        run(main())
    assert exc.value.status_code == 401 and "log in first" in exc.value.message
