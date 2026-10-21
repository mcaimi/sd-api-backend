#!/usr/bin/env python
"""
Outputs Router

Endpoints for browsing and downloading persisted generation outputs
(image + metadata pairs saved by api/services/generation.py:persist_to_backend).
"""

import json
import re
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.models.outputs import OutputItem, OutputListResponse
from libs.shared.config import get_app_config

router = APIRouter()

# Outputs are saved under a uuid4 stem shared by the .png and .json files.
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


def _validate_id(output_id: str) -> str:
    if not UUID_RE.match(output_id):
        raise HTTPException(status_code=400, detail="Invalid output id")
    return output_id


@router.get("/", response_model=OutputListResponse)
async def list_outputs(limit: int = 24, offset: int = 0):
    """List persisted generation outputs, newest first."""
    config = get_app_config()
    images_dir = Path(config.output_images_path)
    json_dir = Path(config.output_json_path)

    image_stems = {p.stem: p for p in images_dir.glob("*.png")}
    json_stems = {p.stem: p for p in json_dir.glob("*.json")}
    paired_ids = image_stems.keys() & json_stems.keys()

    entries = sorted(
        (image_stems[stem] for stem in paired_ids),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    total = len(entries)
    page = entries[offset : offset + limit]

    items = [
        OutputItem(
            id=image_path.stem,
            created_at=datetime.fromtimestamp(image_path.stat().st_mtime).isoformat(),
            size_bytes=image_path.stat().st_size,
            image_url=f"/api/v1/outputs/{image_path.stem}/image",
            json_url=f"/api/v1/outputs/{image_path.stem}/metadata",
        )
        for image_path in page
    ]

    return OutputListResponse(total=total, limit=limit, offset=offset, items=items)


@router.get("/{output_id}/image")
async def get_output_image(output_id: str):
    """Download the persisted PNG for a generation output."""
    output_id = _validate_id(output_id)
    config = get_app_config()
    image_path = Path(config.output_images_path) / f"{output_id}.png"

    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Output image not found")

    return FileResponse(image_path, media_type="image/png")


@router.get("/{output_id}/metadata")
async def get_output_metadata(output_id: str):
    """Fetch the persisted generation metadata JSON for an output."""
    output_id = _validate_id(output_id)
    config = get_app_config()
    json_path = Path(config.output_json_path) / f"{output_id}.json"

    if not json_path.is_file():
        raise HTTPException(status_code=404, detail="Output metadata not found")

    with open(json_path) as f:
        return json.load(f)


@router.delete("/{output_id}", status_code=204)
async def delete_output(output_id: str):
    """Delete the persisted image and metadata for a generation output."""
    output_id = _validate_id(output_id)
    config = get_app_config()
    image_path = Path(config.output_images_path) / f"{output_id}.png"
    json_path = Path(config.output_json_path) / f"{output_id}.json"

    if not image_path.is_file() and not json_path.is_file():
        raise HTTPException(status_code=404, detail="Output not found")

    image_path.unlink(missing_ok=True)
    json_path.unlink(missing_ok=True)
