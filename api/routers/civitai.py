#!/usr/bin/env python
"""
Civitai Router

Search Civitai and inspect models/files before downloading them into the
configured model directories. Downloads run as "io" lane jobs, tracked
through the /jobs endpoints.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.config import get_app_config
from api.constants import MODEL_TYPES, RESOURCE_TYPES, validate_model_type, validate_resource_type
from api.models.civitai import (
    CivitaiDownloadRequest,
    CivitaiDownloadResponse,
    CivitaiIdentifyRequest,
    CivitaiIdentifyResponse,
    CivitaiLayoutResponse,
    CivitaiModelDetail,
    CivitaiSearchResponse,
    CivitaiStatusResponse,
)
from api.services import civitai_mapping as mapping
from api.services.civitai_client import CivitaiClient, CivitaiError, get_civitai_client
from api.services import model_layout
from api.services.downloads import DownloadError, build_sidecar, save_preview, sidecar_preview_url, start_download
from api.services.model_files import (
    LocalModelIndex,
    preview_path,
    read_sidecar,
    sanitize_filename,
    sha256_file,
    write_sidecar,
)
from metastable.models import enumerate_models

logger = logging.getLogger(__name__)

router = APIRouter()


def require_enabled() -> CivitaiClient:
    """Dependency: the Civitai client, or 404 when the feature is disabled."""
    if not get_app_config().civitai_enabled:
        raise HTTPException(status_code=404, detail="Civitai integration is disabled (civitai.enabled)")
    return get_civitai_client()


def civitai_http_error(error: CivitaiError) -> HTTPException:
    headers = {"Retry-After": error.retry_after} if error.retry_after else None
    return HTTPException(status_code=error.status_code, detail=error.message, headers=headers)


@router.get("/status", response_model=CivitaiStatusResponse)
async def civitai_status():
    """Feature status and the type mappings used for Civitai filters."""
    config = get_app_config()
    return CivitaiStatusResponse(
        enabled=config.civitai_enabled,
        has_api_key=bool(config.civitai_api_key),
        nsfw_default=config.civitai_nsfw,
        model_types={t: mapping.base_models(t) for t in MODEL_TYPES},
        resource_types={t: mapping.civitai_types(t) for t in RESOURCE_TYPES},
        sort_options=list(mapping.SORT_OPTIONS),
        period_options=list(mapping.PERIOD_OPTIONS),
    )


@router.get("/search", response_model=CivitaiSearchResponse)
async def search_models(
    model_type: str,
    resource_type: str,
    query: Optional[str] = Query(None, max_length=200),
    sort: str = mapping.SORT_OPTIONS[0],
    period: str = mapping.PERIOD_OPTIONS[0],
    nsfw: Optional[bool] = None,
    limit: int = Query(24, ge=1, le=100),
    cursor: Optional[str] = Query(None, max_length=200),
    client: CivitaiClient = Depends(require_enabled),
):
    """
    Search Civitai for models usable as model_type/resource_type.

    Paginate by passing the returned next_cursor back as cursor.
    """
    validate_model_type(model_type)
    validate_resource_type(resource_type)
    if sort not in mapping.SORT_OPTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid sort. Must be one of {list(mapping.SORT_OPTIONS)}")
    if period not in mapping.PERIOD_OPTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid period. Must be one of {list(mapping.PERIOD_OPTIONS)}")
    nsfw_allowed = get_app_config().civitai_nsfw if nsfw is None else nsfw

    base_models = mapping.base_models(model_type, await client.known_base_models())
    if not base_models:
        return CivitaiSearchResponse(items=[])

    try:
        data = await client.search(
            types=mapping.civitai_types(resource_type),
            base_models=base_models,
            query=query.strip() if query else None,
            sort=sort,
            period=period,
            nsfw=nsfw_allowed,
            limit=limit,
            cursor=cursor,
        )
    except CivitaiError as e:
        raise civitai_http_error(e)

    items = [mapping.normalize_summary(model, model_type, nsfw_allowed) for model in data.get("items") or []]
    return CivitaiSearchResponse(
        items=[item for item in items if item],
        next_cursor=(data.get("metadata") or {}).get("nextCursor"),
    )


async def _mark_local_files(detail: dict, model_type: str) -> None:
    """Fill local_path/local_match for files already present locally."""
    paths = get_app_config().get_model_paths(model_type)
    indexes: Dict[str, LocalModelIndex] = {}
    for version in detail["versions"]:
        for file in version["files"]:
            target = file["target_resource_type"]
            if target not in paths:
                continue
            if target not in indexes:
                indexes[target] = await asyncio.to_thread(LocalModelIndex, paths[target])
            path, match = indexes[target].lookup(sanitize_filename(file["name"]), file["sha256"])
            if path:
                file["local_path"] = str(path)
                file["local_match"] = match


@router.get("/models/{model_id}", response_model=CivitaiModelDetail)
async def get_model(
    model_id: int,
    model_type: str,
    nsfw: Optional[bool] = None,
    client: CivitaiClient = Depends(require_enabled),
):
    """
    A Civitai model with its versions and files. Each file says whether it
    can be downloaded for model_type (and why not), where it would go, and
    whether it's already present locally.
    """
    validate_model_type(model_type)
    nsfw_allowed = get_app_config().civitai_nsfw if nsfw is None else nsfw
    try:
        model = await client.get_model(model_id)
    except CivitaiError as e:
        raise civitai_http_error(e)

    detail = mapping.normalize_detail(model, model_type, nsfw_allowed)
    await _mark_local_files(detail, model_type)
    return detail


@router.post("/downloads", response_model=CivitaiDownloadResponse, status_code=202)
async def download_file(request: CivitaiDownloadRequest, client: CivitaiClient = Depends(require_enabled)):
    """
    Download a Civitai file into the model directories as a background job.

    Track it (progress, cancel) through /jobs/{job_id}. A base model that
    doesn't match model_type answers 409 with warnings; resend with
    accept_mismatch=true to download anyway.
    """
    validate_model_type(request.model_type)
    validate_resource_type(request.resource_type)
    try:
        return await start_download(
            version_id=request.version_id,
            file_id=request.file_id,
            model_type=request.model_type,
            resource_type=request.resource_type,
            accept_mismatch=request.accept_mismatch,
            client=client,
        )
    except DownloadError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


# Flux layouts of remote files, by SHA256 (a file's content never changes)
_layout_cache: Dict[str, dict] = {}


@router.get("/files/layout", response_model=CivitaiLayoutResponse)
async def file_layout(version_id: int, file_id: int, client: CivitaiClient = Depends(require_enabled)):
    """
    Which components a Flux checkpoint on Civitai bundles (transformer,
    CLIP-L, T5-XXL, VAE), read from its safetensors header without
    downloading it, and which shared components it would need from
    flux.components_path that are missing there. Most Flux checkpoints need
    an API key even for this.
    """
    try:
        version = await client.get_version(version_id)
    except CivitaiError as e:
        raise civitai_http_error(e)
    file = next((f for f in version.get("files") or [] if f.get("id") == file_id), None)
    if file is None:
        raise HTTPException(status_code=404, detail=f"File {file_id} is not part of Civitai version {version_id}")
    if mapping.model_type_for_base_model(version.get("baseModel")) != "flux":
        raise HTTPException(status_code=400, detail="Layout checks are only available for Flux models")
    if (file.get("metadata") or {}).get("format") != "SafeTensor":
        raise HTTPException(status_code=400, detail="Only safetensors files have a readable layout")

    sha256 = ((file.get("hashes") or {}).get("SHA256") or "").lower()
    layout = _layout_cache.get(sha256) if sha256 else None
    if layout is None:
        try:
            header = await client.peek_header(file.get("downloadUrl") or "")
        except CivitaiError as e:
            raise civitai_http_error(e)
        layout = model_layout.classify_keys(header.keys())
        if sha256:
            _layout_cache[sha256] = layout

    missing = []
    if layout["transformer"]:
        missing = model_layout.missing_components(layout, get_app_config().flux_components_path)
    return CivitaiLayoutResponse(layout=layout, missing_components=missing)


@router.post("/identify", response_model=CivitaiIdentifyResponse)
async def identify_model(request: CivitaiIdentifyRequest, client: CivitaiClient = Depends(require_enabled)):
    """
    Look up a local model on Civitai by its SHA256 and, on a match, write
    its .civitai.json sidecar (and preview). Hashing a large checkpoint
    takes a while; the hash is kept in the sidecar so it's computed once.
    """
    validate_model_type(request.model_type)
    validate_resource_type(request.resource_type)
    models_dir = get_app_config().get_model_paths(request.model_type)[request.resource_type]
    model_path = enumerate_models(models_dir).get(request.model_name)
    if not model_path:
        raise HTTPException(status_code=404, detail="Model not found")
    model_path = Path(model_path)

    sidecar = read_sidecar(model_path) or {}
    sha256 = sidecar.get("sha256") or await asyncio.to_thread(sha256_file, model_path)
    try:
        version = await client.get_version_by_hash(sha256)
    except CivitaiError as e:
        if e.status_code != 404:
            raise civitai_http_error(e)
        if not sidecar:
            # Remember the hash so the next attempt doesn't rehash the file
            await asyncio.to_thread(write_sidecar, model_path, {"source": "local", "sha256": sha256})
        return CivitaiIdentifyResponse(matched=False)

    files = version.get("files") or []
    file = next(
        (f for f in files if ((f.get("hashes") or {}).get("SHA256") or "").lower() == sha256),
        next((f for f in files if f.get("primary")), files[0] if files else {}),
    )
    sidecar = build_sidecar(version, file, sha256, model_path.stat().st_size)
    sidecar["identified_at"] = datetime.now().isoformat(timespec="seconds")
    if request.model_type == "flux" and request.resource_type == "checkpoints":
        try:
            sidecar["flux_layout"] = await asyncio.to_thread(model_layout.classify, model_path)
        except Exception as e:
            logger.warning(f"Could not read the Flux layout of {model_path}: {e}")
    await asyncio.to_thread(write_sidecar, model_path, sidecar)

    preview_url = sidecar_preview_url(version)
    if preview_url and not preview_path(model_path).exists():
        await save_preview(client, preview_url, preview_path(model_path))
    return CivitaiIdentifyResponse(matched=True, sidecar=sidecar)
