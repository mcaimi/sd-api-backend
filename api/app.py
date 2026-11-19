#!/usr/bin/env python
"""
FastAPI Application Factory

Creates and configures the FastAPI application with all routers and middleware.
"""

import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

import metastable

from api.routers import models, jobs, system, info, merge, outputs, civitai
from api.routers.generation import ARCHITECTURES, create_generation_router
from api.services.job_queue import get_job_queue
from api.services.civitai_client import close_civitai_client
from api.services.downloads import cleanup_partial_downloads
from api.config import get_app_config

WEBUI_DIR = Path(__file__).resolve().parent.parent / "webui"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Starting SD API Backend...")
    cleanup_partial_downloads()
    job_queue = get_job_queue()
    await job_queue.start_workers()
    logger.info("Job queue workers started")

    yield

    # Shutdown
    logger.info("Shutting down SD API Backend...")
    await job_queue.stop_workers()
    logger.info("Job queue workers stopped")
    await close_civitai_client()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance
    """
    # Load configuration
    config = get_app_config()

    # Register the compute device before any torch.cuda usage, so ROCm
    # environment overrides (gfx_override, env) take effect
    metastable.configure_device(config.device)

    app = FastAPI(
        title="Stable Diffusion API",
        description="REST API for SD1.5, SDXL, Illustrious XL, Flux, and Z-Image image generation, inpainting, and model merging",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS middleware
    if config.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # A model type that parameters.yaml doesn't configure (e.g. zimage in a
    # config written before it was supported) is a client-visible 400
    @app.exception_handler(metastable.ConfigurationError)
    async def configuration_error_handler(request: Request, exc: metastable.ConfigurationError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # Include routers
    for arch in ARCHITECTURES:
        app.include_router(
            create_generation_router(arch),
            prefix=f"/api/v1/{arch.model_type}",
            tags=[arch.display_name],
        )
    app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
    app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["Jobs"])
    app.include_router(system.router, prefix="/api/v1/system", tags=["System"])
    app.include_router(info.router, prefix="/api/v1/info", tags=["Informational"])
    app.include_router(merge.router, prefix="/api/v1/merge", tags=["Merge"])
    app.include_router(outputs.router, prefix="/api/v1/outputs", tags=["Outputs"])
    app.include_router(civitai.router, prefix="/api/v1/civitai", tags=["Civitai"])

    # Management web UI (static, no build step)
    if WEBUI_DIR.is_dir():
        app.mount("/ui", StaticFiles(directory=WEBUI_DIR, html=True), name="ui")

    logger.info("FastAPI application created")

    return app


# Create app instance
app = create_app()
