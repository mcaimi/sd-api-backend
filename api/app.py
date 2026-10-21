#!/usr/bin/env python
"""
FastAPI Application Factory

Creates and configures the FastAPI application with all routers and middleware.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routers import sd15, sdxl, illustrious, flux, models, jobs, system, info, merge
from api.services.job_queue import get_job_queue
from libs.shared.config import get_app_config

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
    job_queue = get_job_queue()
    await job_queue.start_workers()
    logger.info("Job queue workers started")

    yield

    # Shutdown
    logger.info("Shutting down SD API Backend...")
    await job_queue.stop_workers()
    logger.info("Job queue workers stopped")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance
    """
    # Load configuration
    config = get_app_config()

    app = FastAPI(
        title="Stable Diffusion API",
        description="REST API for SD1.5, SDXL, Illustrious XL, and Flux image generation, inpainting, and model merging",
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

    # Include routers
    app.include_router(sd15.router, prefix="/api/v1/sd15", tags=["SD1.5"])
    app.include_router(sdxl.router, prefix="/api/v1/sdxl", tags=["SDXL"])
    app.include_router(illustrious.router, prefix="/api/v1/illustrious", tags=["Illustrious"])
    app.include_router(flux.router, prefix="/api/v1/flux", tags=["Flux"])
    app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
    app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["Jobs"])
    app.include_router(system.router, prefix="/api/v1/system", tags=["System"])
    app.include_router(info.router, prefix="/api/v1/info", tags=["Informational"])
    app.include_router(merge.router, prefix="/api/v1/merge", tags=["Merge"])

    logger.info("FastAPI application created")

    return app


# Create app instance
app = create_app()
