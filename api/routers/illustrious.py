#!/usr/bin/env python
"""
Illustrious XL Router

Endpoints for Illustrious XL image generation and inpainting.
"""

from fastapi import APIRouter
from pathlib import Path

from api.models.generation import (
    GenerationRequest,
    InpaintRequest,
    BatchGenerationRequest,
    CompareModelsRequest,
    JobResponse,
)
from api.services.generation import generate_image_async, inpaint_image_async
from api.services.job_queue import get_job_queue
from libs.shared.config import get_app_config

router = APIRouter()


@router.post("/generate", response_model=JobResponse)
async def generate_illustrious(request: GenerationRequest):
    """Generate an Illustrious XL image (async job)."""
    config = get_app_config()
    model_path = Path(config.checkpoints_illustrious_path) / request.model_checkpoint

    # Convert LoraConfig to dict
    loras_dict = None
    if request.loras:
        loras_dict = [
            {
                "name": Path(config.loras_illustrious_path) / lora.name,
                "strength": lora.strength,
            }
            for lora in request.loras
        ]

    # Submit to job queue
    job_queue = get_job_queue()
    job_id = await job_queue.submit_job(
        job_type="illustrious_generation",
        params={
            "model_type": "illustrious",
            "model_path": str(model_path),
            "positive_prompt": request.positive_prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "cfg": request.cfg_scale,
            "seed": request.seed,
            "scheduler": request.scheduler,
            "loras": loras_dict,
            "custom_vae": request.custom_vae,
        },
        callback=generate_image_async,
    )

    return JobResponse(
        job_id=job_id, status="queued", message="Generation job submitted successfully"
    )


@router.post("/inpaint", response_model=JobResponse)
async def inpaint_illustrious(request: InpaintRequest):
    """Illustrious XL inpainting (async job)."""
    config = get_app_config()
    model_path = Path(config.checkpoints_illustrious_path) / request.model_checkpoint

    # Convert LoraConfig to dict
    loras_dict = None
    if request.loras:
        loras_dict = [
            {
                "name": Path(config.loras_illustrious_path) / lora.name,
                "strength": lora.strength,
            }
            for lora in request.loras
        ]

    # Submit to job queue
    job_queue = get_job_queue()
    job_id = await job_queue.submit_job(
        job_type="illustrious_inpaint",
        params={
            "model_type": "illustrious",
            "model_path": str(model_path),
            "positive_prompt": request.positive_prompt,
            "negative_prompt": request.negative_prompt,
            "image_base64": request.image_base64,
            "mask_base64": request.mask_base64,
            "steps": request.steps,
            "cfg": request.cfg_scale,
            "seed": request.seed,
            "scheduler": request.scheduler,
            "loras": loras_dict,
            "custom_vae": request.custom_vae,
        },
        callback=inpaint_image_async,
    )

    return JobResponse(
        job_id=job_id, status="queued", message="Inpainting job submitted successfully"
    )


@router.post("/generate/batch")
async def batch_generate_illustrious(request: BatchGenerationRequest):
    """Generate multiple Illustrious XL images."""
    job_ids = []

    for gen_request in request.requests:
        response = await generate_illustrious(gen_request)
        job_ids.append(response.job_id)

    return {"job_ids": job_ids, "message": f"Submitted {len(job_ids)} generation jobs"}


@router.post("/generate/compare")
async def compare_models_illustrious(request: CompareModelsRequest):
    """Generate images from multiple Illustrious XL models for comparison."""
    job_ids = []

    for model in request.models:
        gen_request = GenerationRequest(
            positive_prompt=request.positive_prompt,
            negative_prompt=request.negative_prompt,
            model_checkpoint=model,
            width=request.width,
            height=request.height,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            scheduler=request.scheduler,
        )
        response = await generate_illustrious(gen_request)
        job_ids.append({"model": model, "job_id": response.job_id})

    return {
        "comparisons": job_ids,
        "message": f"Submitted {len(job_ids)} comparison jobs",
    }
