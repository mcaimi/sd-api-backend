"""
Info Router

Endpoints for informational requests.
"""

from typing import List
from fastapi import APIRouter, HTTPException

from api.config import get_app_config
from metastable.models import enumerate_models
from metastable.schedulers import SCHEDULERS

router = APIRouter()

@router.get("/schedulers", response_model=List[str])
async def list_schedulers():
    """
    List available noise schedulers
    """

    return [
        scheduler_name for scheduler_name in SCHEDULERS.keys()
    ]


@router.get("/app_settings", response_model=dict)
async def list_config():
    """
    Return a list of config parameters
    """
    
    return get_app_config().as_dict()