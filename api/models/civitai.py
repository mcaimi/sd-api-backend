#!/usr/bin/env python
"""
Pydantic Models for Civitai Endpoints

Normalised views of Civitai search results, models, versions and files,
plus download and layout requests/responses.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CivitaiPreview(BaseModel):
    url: str
    type: str  # "image" or "video"
    nsfw_level: int = 0
    width: Optional[int] = None
    height: Optional[int] = None


class CivitaiStats(BaseModel):
    downloads: int = 0
    rating: Optional[float] = None
    rating_count: int = 0
    thumbs_up: int = 0
    favorites: int = 0


class CivitaiFile(BaseModel):
    id: int
    name: str
    size_bytes: int
    type: Optional[str] = None
    format: Optional[str] = None
    fp: Optional[str] = None
    size_variant: Optional[str] = None
    sha256: Optional[str] = None
    primary: bool = False
    target_resource_type: Optional[str] = Field(
        None, description="Resource directory the file would be saved to"
    )
    compatible: bool
    reason: Optional[str] = Field(None, description="Why the file can't be downloaded")
    warnings: List[str] = []
    local_path: Optional[str] = Field(None, description="Already present locally at this path")
    local_match: Optional[str] = Field(None, description="'sha256' or 'name'")


class CivitaiVersion(BaseModel):
    id: int
    name: str
    base_model: Optional[str] = None
    base_model_type: Optional[str] = None
    model_type: Optional[str] = Field(None, description="App model type of the base model")
    trained_words: List[str] = []
    published_at: Optional[str] = None
    early_access: bool = False
    preview: Optional[CivitaiPreview] = None
    files: List[CivitaiFile] = []


class CivitaiModelSummary(BaseModel):
    id: int
    name: str
    type: Optional[str] = None
    creator: Optional[str] = None
    nsfw: bool = False
    tags: List[str] = []
    stats: CivitaiStats
    preview: Optional[CivitaiPreview] = None
    base_models: List[str] = []


class CivitaiModelDetail(CivitaiModelSummary):
    description: str = ""
    versions: List[CivitaiVersion] = []


class CivitaiSearchResponse(BaseModel):
    items: List[CivitaiModelSummary]
    next_cursor: Optional[str] = None


class CivitaiStatusResponse(BaseModel):
    enabled: bool
    has_api_key: bool
    nsfw_default: bool
    model_types: Dict[str, List[str]] = Field(..., description="App model type -> Civitai base models")
    resource_types: Dict[str, List[str]] = Field(..., description="App resource type -> Civitai model types")
    sort_options: List[str]
    period_options: List[str]


class CivitaiDownloadRequest(BaseModel):
    version_id: int
    file_id: int
    model_type: str
    resource_type: str
    accept_mismatch: bool = Field(False, description="Download even if the base model doesn't match model_type")


class CivitaiDownloadResponse(BaseModel):
    job_id: str
    filename: str
    path: str
    warnings: List[str] = []


class FluxLayout(BaseModel):
    transformer: bool
    clip_l: bool
    t5xxl: bool
    vae: bool
    format: Optional[str] = None  # "bfl" or "diffusers"
    variant: Optional[str] = None  # "dev" or "schnell"


class CivitaiLayoutResponse(BaseModel):
    layout: FluxLayout
    missing_components: List[str] = Field(
        [], description="Shared Flux components this file needs but flux.components_path lacks"
    )


class CivitaiIdentifyRequest(BaseModel):
    model_type: str
    resource_type: str
    model_name: str


class CivitaiIdentifyResponse(BaseModel):
    matched: bool
    sidecar: Optional[dict] = None
