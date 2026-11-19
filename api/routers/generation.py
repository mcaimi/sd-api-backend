#!/usr/bin/env python
"""
Generation Routers

Endpoints for image generation and inpainting, one router per architecture.
Every architecture exposes the same endpoints; they differ only in their
request models (defaults, scheduler set, architecture-specific arguments).
"""

from dataclasses import dataclass
from typing import Type

from fastapi import APIRouter

from api.models.generation import (
    GenerationRequest,
    InpaintRequest,
    BatchGenerationRequest,
    CompareModelsRequest,
    FluxGenerationRequest,
    FluxInpaintRequest,
    FluxBatchGenerationRequest,
    FluxCompareModelsRequest,
    JobResponse,
)
from api.services.generation import generate_image_async, inpaint_image_async
from api.services.job_queue import get_job_queue
from api.routers.model_paths import resolve_components_dir, resolve_generation_paths


@dataclass(frozen=True)
class Architecture:
    """An architecture's API name, display name and request models."""

    model_type: str
    display_name: str
    generation_request: Type[GenerationRequest]
    inpaint_request: Type[InpaintRequest]
    batch_request: Type[BatchGenerationRequest]
    compare_request: Type[CompareModelsRequest]


ARCHITECTURES = (
    Architecture("sd15", "SD1.5", GenerationRequest, InpaintRequest,
                 BatchGenerationRequest, CompareModelsRequest),
    Architecture("sdxl", "SDXL", GenerationRequest, InpaintRequest,
                 BatchGenerationRequest, CompareModelsRequest),
    Architecture("illustrious", "Illustrious XL", GenerationRequest, InpaintRequest,
                 BatchGenerationRequest, CompareModelsRequest),
    Architecture("flux", "Flux", FluxGenerationRequest, FluxInpaintRequest,
                 FluxBatchGenerationRequest, FluxCompareModelsRequest),
)


def _common_params(model_type: str, request) -> dict:
    """Job parameters shared by generation and inpainting."""
    model_path, loras_dict, custom_vae = resolve_generation_paths(model_type, request)
    return {
        "model_type": model_type,
        "model_path": str(model_path),
        "positive_prompt": request.positive_prompt,
        "negative_prompt": request.negative_prompt,
        "steps": request.steps,
        "cfg": request.cfg_scale,
        "seed": request.seed,
        "scheduler": request.scheduler,
        "loras": loras_dict,
        "custom_vae": custom_vae,
        "components_dir": resolve_components_dir(model_type),
        "pipeline_kwargs": request.pipeline_kwargs(),
    }


def create_generation_router(arch: Architecture) -> APIRouter:
    """Build the generate/inpaint/batch/compare endpoints of an architecture."""
    router = APIRouter()
    model_type, name = arch.model_type, arch.display_name

    async def generate(request: arch.generation_request) -> JobResponse:
        params = _common_params(model_type, request)
        params.update(width=request.width, height=request.height)

        job_id = await get_job_queue().submit_job(
            job_type=f"{model_type}_generation",
            params=params,
            callback=generate_image_async,
        )
        return JobResponse(
            job_id=job_id, status="queued", message="Generation job submitted successfully"
        )

    async def inpaint(request: arch.inpaint_request) -> JobResponse:
        params = _common_params(model_type, request)
        params.update(image_base64=request.image_base64, mask_base64=request.mask_base64)

        job_id = await get_job_queue().submit_job(
            job_type=f"{model_type}_inpaint",
            params=params,
            callback=inpaint_image_async,
        )
        return JobResponse(
            job_id=job_id, status="queued", message="Inpainting job submitted successfully"
        )

    async def batch_generate(request: arch.batch_request):
        job_ids = []
        for gen_request in request.requests:
            response = await generate(gen_request)
            job_ids.append(response.job_id)

        return {"job_ids": job_ids, "message": f"Submitted {len(job_ids)} generation jobs"}

    async def compare_models(request: arch.compare_request):
        job_ids = []
        settings = request.model_dump(exclude={"models"})
        for model in request.models:
            gen_request = arch.generation_request(model_checkpoint=model, **settings)
            response = await generate(gen_request)
            job_ids.append({"model": model, "job_id": response.job_id})

        return {
            "comparisons": job_ids,
            "message": f"Submitted {len(job_ids)} comparison jobs",
        }

    router.add_api_route(
        "/generate", generate, methods=["POST"], response_model=JobResponse,
        name=f"generate_{model_type}", summary=f"Generate a {name} image (async job)",
    )
    router.add_api_route(
        "/inpaint", inpaint, methods=["POST"], response_model=JobResponse,
        name=f"inpaint_{model_type}", summary=f"{name} inpainting (async job)",
    )
    router.add_api_route(
        "/generate/batch", batch_generate, methods=["POST"],
        name=f"batch_generate_{model_type}", summary=f"Generate multiple {name} images",
    )
    router.add_api_route(
        "/generate/compare", compare_models, methods=["POST"],
        name=f"compare_models_{model_type}",
        summary=f"Generate images from multiple {name} models for comparison",
    )
    return router
