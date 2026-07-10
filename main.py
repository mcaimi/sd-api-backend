#!/usr/bin/env python
"""
SD API Backend Entry Point

Run with: uvicorn main:app --reload
"""

from api.app import app
from libs.shared.config import get_app_config

if __name__ == "__main__":
    import uvicorn

    # Load configuration
    config = get_app_config()

    uvicorn.run(
        "api.app:app",
        host=config.api_host,
        port=config.api_port,
        reload=True,
        log_level="info"
    )
