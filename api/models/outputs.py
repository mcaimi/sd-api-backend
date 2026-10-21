#!/usr/bin/env python
"""
Pydantic Models for Output Browsing Endpoints

Request and response models for listing persisted generation outputs.
"""

from pydantic import BaseModel
from typing import List


class OutputItem(BaseModel):
    """A single persisted generation (image + metadata pair)."""
    id: str
    created_at: str
    size_bytes: int
    image_url: str
    json_url: str


class OutputListResponse(BaseModel):
    """Paginated list of persisted generation outputs."""
    total: int
    limit: int
    offset: int
    items: List[OutputItem]
