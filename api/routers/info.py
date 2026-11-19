"""
Info Router

Endpoints for informational requests.
"""

from typing import List, Optional
from fastapi import APIRouter

from api.config import get_app_config
from api.constants import validate_model_type
from metastable.pipelines import get_generator_class
from metastable.schedulers import SCHEDULERS

router = APIRouter()

@router.get("/schedulers", response_model=List[str])
async def list_schedulers(model_type: Optional[str] = None):
    """
    List available noise schedulers

    Without model_type, lists the SD1.5/SDXL/Illustrious schedulers; Flux and
    Z-Image use the flow-matching set (model_type=flux or zimage).
    """
    if model_type is None:
        return list(SCHEDULERS)
    validate_model_type(model_type)
    return list(get_generator_class(model_type).SCHEDULERS)


@router.get("/app_settings", response_model=dict)
async def list_config():
    """
    Return a list of config parameters
    """
    
    return get_app_config().as_dict()