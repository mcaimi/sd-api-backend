#!/usr/bin/env python
"""
Civitai Download Service

Validates a requested Civitai file, then downloads it as a cancellable "io"
lane job: streamed to <models_dir>/civitai/<name>.safetensors.part, hashed
while streaming, verified against Civitai's SHA256 and only then renamed to
its final name, so enumerate_models() never lists a partial file. A
.civitai.json sidecar and a .preview.jpg are written next to the model.
"""

import asyncio
import hashlib
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional
from urllib.parse import urlparse

import httpx

from api.config import get_app_config
from api.constants import MODEL_TYPES, RESOURCE_TYPES
from api.services import civitai_mapping as mapping
from api.services.civitai_client import (
    CivitaiClient,
    CivitaiError,
    get_civitai_client,
    is_download_url,
    raise_for_civitai_status,
)
from api.services.job_queue import IO_LANE, get_job_queue
from api.services.model_files import (
    MODEL_SUFFIX,
    PARTIAL_SUFFIX,
    LocalModelIndex,
    is_within,
    partial_path,
    preview_path,
    sanitize_filename,
    write_sidecar,
)
from api.services import model_layout

logger = logging.getLogger(__name__)

JOB_TYPE = "civitai_download"
DOWNLOAD_SUBDIR = "civitai"
WRITE_BUFFER_BYTES = 8 * 1024 * 1024
REPORT_INTERVAL = 0.5  # seconds between progress updates
DOWNLOAD_TIMEOUT = httpx.Timeout(connect=30, read=120, write=30, pool=30)
PREVIEW_WIDTH = 450
PREVIEW_HOST_SUFFIX = ".civitai.com"


class DownloadError(Exception):
    """A download request that can't be started; detail is returned as the HTTP error detail."""

    def __init__(self, status_code: int, detail):
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail


def _gb(size_bytes: float) -> str:
    return f"{size_bytes / 1e9:.1f} GB"


def build_sidecar(version: dict, file: dict, sha256: str, size_bytes: int) -> dict:
    """The .civitai.json sidecar describing a model file downloaded from (or matched on) Civitai."""
    model = version.get("model") or {}
    return {
        "source": "civitai",
        "model_id": version.get("modelId"),
        "model_name": model.get("name"),
        "civitai_type": model.get("type"),
        "version_id": version.get("id"),
        "version_name": version.get("name"),
        "file_id": file.get("id"),
        "file_name": file.get("name"),
        "base_model": version.get("baseModel"),
        "trained_words": version.get("trainedWords") or [],
        "nsfw": bool(model.get("nsfw")),
        "sha256": sha256,
        "size_bytes": size_bytes,
        "url": f"https://civitai.com/models/{version.get('modelId')}?modelVersionId={version.get('id')}",
    }


def sidecar_preview_url(version: dict) -> Optional[str]:
    """URL of a SFW still image of the version, to save as the model's preview."""
    images = [i for i in version.get("images") or [] if (i.get("type") or "image") == "image"]
    preview = mapping.pick_preview(images, nsfw_allowed=False)
    return preview["url"] if preview else None


async def start_download(
    version_id: int,
    file_id: int,
    model_type: str,
    resource_type: str,
    accept_mismatch: bool = False,
    client: Optional[CivitaiClient] = None,
) -> dict:
    """
    Validate a Civitai file and submit its download job.

    Returns:
        {"job_id", "filename", "path", "warnings"}

    Raises:
        DownloadError: with the HTTP status and detail to return
    """
    client = client or get_civitai_client()
    config = get_app_config()
    if model_type not in MODEL_TYPES or resource_type not in RESOURCE_TYPES:
        raise DownloadError(400, "Invalid model_type or resource_type")

    try:
        version = await client.get_version(version_id)
    except CivitaiError as e:
        raise DownloadError(e.status_code, e.message)

    file = next((f for f in version.get("files") or [] if f.get("id") == file_id), None)
    if file is None:
        raise DownloadError(404, f"File {file_id} is not part of Civitai version {version_id}")

    model = version.get("model") or {}
    compatibility = mapping.file_compatibility(file, version, model.get("type"), model_type)
    if not compatibility.ok:
        raise DownloadError(400, compatibility.reason)
    if compatibility.target_resource_type != resource_type:
        raise DownloadError(400, f"This file belongs in {compatibility.target_resource_type}, not {resource_type}")
    mismatch = [w for w in compatibility.warnings if w.startswith("Made for ")]
    if mismatch and not accept_mismatch:
        raise DownloadError(409, {
            "message": "Base model doesn't match; resend with accept_mismatch to download anyway",
            "warnings": mismatch,
        })

    url = file.get("downloadUrl") or ""
    if not is_download_url(url):
        raise DownloadError(502, f"Unexpected download URL from Civitai: {url}")
    sha256 = file["hashes"]["SHA256"].lower()
    size_bytes = int(round((file.get("sizeKB") or 0) * 1024))

    models_dir = Path(config.get_model_paths(model_type)[resource_type])
    dest_dir = models_dir / DOWNLOAD_SUBDIR
    queue = get_job_queue()

    active = await queue.find_active_job(JOB_TYPE, file_id=file_id)
    if active:
        raise DownloadError(409, {"message": "This file is already being downloaded", "job_id": active})

    # File names must be unique across the whole resource tree (enumerate_models keys by name)
    index = await asyncio.to_thread(LocalModelIndex, str(models_dir))
    filename = sanitize_filename(file["name"])
    existing, match = index.lookup(filename, sha256)
    if match == "sha256":
        raise DownloadError(409, {"message": "Already downloaded", "path": str(existing)})
    if match == "name":
        filename = sanitize_filename(f"{Path(filename).stem}-v{version_id}")
        if index.lookup(filename)[0]:
            raise DownloadError(409, {"message": f"A different {filename} already exists", "path": str(existing)})
    dest = dest_dir / filename
    if not is_within(dest, models_dir):
        raise DownloadError(400, "Refusing to write outside the models directory")
    if await queue.find_active_job(JOB_TYPE, dest=str(dest)):
        raise DownloadError(409, {"message": f"Another download is already writing {filename}"})

    dest_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest_dir).free
    reserve = config.civitai_min_free_space_gb * 1e9
    if free - size_bytes < reserve:
        raise DownloadError(507, (
            f"Not enough disk space: {filename} needs {_gb(size_bytes)}, {_gb(free)} free, "
            f"and civitai.min_free_space_gb keeps {_gb(reserve)} free"
        ))

    sidecar = build_sidecar(version, file, sha256, size_bytes)
    preview_url = sidecar_preview_url(version)
    job_id = await queue.submit_job(
        JOB_TYPE,
        {
            "url": url,
            "dest": str(dest),
            "sha256": sha256,
            "size_bytes": size_bytes,
            "file_id": file_id,
            "model_type": model_type,
            "resource_type": resource_type,
            "sidecar": sidecar,
            "preview_url": preview_url,
        },
        _download,
        lane=IO_LANE,
        cancellable=True,
        with_reporter=True,
    )
    logger.info(f"Queued Civitai download of {file['name']} (version {version_id}) to {dest}")
    return {"job_id": job_id, "filename": filename, "path": str(dest), "warnings": compatibility.warnings}


def _write_and_hash(handle, hasher, data: bytes) -> None:
    handle.write(data)
    hasher.update(data)


async def _download(
    url: str,
    dest: str,
    sha256: str,
    size_bytes: int,
    file_id: int,
    model_type: str,
    resource_type: str,
    sidecar: dict,
    preview_url: Optional[str],
    report: Callable,
) -> dict:
    """Job callback: stream, verify, commit and annotate one download."""
    client = get_civitai_client()
    dest_path = Path(dest)
    part = partial_path(dest_path)
    hasher = hashlib.sha256()
    done = 0

    try:
        async with client.http.stream("GET", url, headers=client.auth_headers(), timeout=DOWNLOAD_TIMEOUT) as response:
            if response.status_code >= 400:
                await response.aread()
                raise_for_civitai_status(response, client.has_api_key)
            content_type = response.headers.get("content-type", "")
            if content_type.startswith(("text/html", "application/json")):
                raise RuntimeError(
                    "Civitai returned a web page instead of the file (login or early access may be required)"
                )
            total = int(response.headers.get("content-length") or 0) or size_bytes

            started = last_report = time.monotonic()
            buffer = bytearray()

            def progress(phase: str) -> None:
                elapsed = max(time.monotonic() - started, 1e-6)
                speed = done / elapsed
                report(done / total if total else 0.0, {
                    "phase": phase,
                    "filename": dest_path.name,
                    "bytes_done": done,
                    "bytes_total": total,
                    "speed_bps": round(speed),
                    "eta_s": round((total - done) / speed) if speed and total > done else None,
                })

            progress("downloading")
            with open(part, "wb") as handle:
                async for chunk in response.aiter_bytes():
                    buffer.extend(chunk)
                    done += len(chunk)
                    if len(buffer) >= WRITE_BUFFER_BYTES:
                        await asyncio.to_thread(_write_and_hash, handle, hasher, bytes(buffer))
                        buffer.clear()
                    if time.monotonic() - last_report >= REPORT_INTERVAL:
                        progress("downloading")
                        last_report = time.monotonic()
                if buffer:
                    await asyncio.to_thread(_write_and_hash, handle, hasher, bytes(buffer))

        progress("verifying")
        digest = hasher.hexdigest()
        if digest != sha256:
            raise RuntimeError(f"Hash mismatch: expected {sha256}, got {digest}; download discarded")
        os.replace(part, dest_path)
    except CivitaiError as e:
        raise RuntimeError(e.message) from e
    finally:
        part.unlink(missing_ok=True)

    sidecar = dict(sidecar, downloaded_at=datetime.now().isoformat(timespec="seconds"))
    result = {
        "path": str(dest_path),
        "filename": dest_path.name,
        "model_type": model_type,
        "resource_type": resource_type,
        "sha256": sha256,
        "size_bytes": done,
        "civitai_model_id": sidecar.get("model_id"),
        "civitai_version_id": sidecar.get("version_id"),
    }
    if model_type == "flux" and resource_type == "checkpoints":
        try:
            layout = await asyncio.to_thread(model_layout.classify, dest_path)
            sidecar["flux_layout"] = result["flux_layout"] = layout
        except Exception as e:
            logger.warning(f"Could not read the Flux layout of {dest_path}: {e}")
    try:
        await asyncio.to_thread(write_sidecar, dest_path, sidecar)
    except OSError as e:
        logger.warning(f"Could not write the Civitai sidecar for {dest_path}: {e}")
    if preview_url:
        await save_preview(client, preview_url, preview_path(dest_path))

    logger.info(f"Downloaded {dest_path} ({_gb(done)})")
    return result


def preview_download_url(url: str) -> Optional[str]:
    """A small rendition of a Civitai image URL, or None if it isn't a Civitai image."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not (parsed.hostname or "").endswith(PREVIEW_HOST_SUFFIX):
        return None
    return url.replace("/original=true/", f"/width={PREVIEW_WIDTH}/")


async def save_preview(client: CivitaiClient, url: str, target: Path) -> None:
    """Best effort: a missing preview never fails the download."""
    url = preview_download_url(url)
    if not url:
        return
    try:
        response = await client.http.get(url)
        response.raise_for_status()
        await asyncio.to_thread(target.write_bytes, response.content)
    except (httpx.HTTPError, OSError) as e:
        logger.warning(f"Could not save preview {target.name}: {e}")


def cleanup_partial_downloads() -> List[Path]:
    """Delete leftover *.safetensors.part files (interrupted downloads) under every model directory."""
    config = get_app_config()
    removed = []
    for model_type in MODEL_TYPES:
        for directory in config.get_model_paths(model_type).values():
            for part in Path(directory).glob(f"**/*{MODEL_SUFFIX}{PARTIAL_SUFFIX}"):
                try:
                    part.unlink()
                    removed.append(part)
                    logger.info(f"Removed partial download {part}")
                except OSError as e:
                    logger.warning(f"Could not remove partial download {part}: {e}")
    return removed
