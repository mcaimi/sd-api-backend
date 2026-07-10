"""
Info Router

Endpoints for informational requests.
"""

from typing import List
from fastapi import APIRouter, HTTPException

from libs.shared.config import get_app_config
from libs.shared.utils import enumerate_models
from libs.globals.vars import schedulers

router = APIRouter()

@router.get("/schedulers", response_model=List[str])
async def list_schedulers():
    """
    List available noise schedulers
    """

    return [
        scheduler_name for scheduler_name in schedulers.keys()
    ]


@router.get("/app_settings", response_model=dict)
async def list_config():
    """
    Return a list of config parameters
    """
    
    return get_app_config().as_dict()