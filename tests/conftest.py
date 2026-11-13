"""Shared fixtures: an AppConfig over temporary model directories."""

import pytest

from api.config import AppConfig
from api.constants import MODEL_TYPES, RESOURCE_TYPES
from api.parameters import Parameters


def make_config(tmp_path, civitai=None, api=None) -> AppConfig:
    sections = {}
    for resource in RESOURCE_TYPES:
        sections[resource] = {}
        for model_type in MODEL_TYPES:
            path = tmp_path / "models" / resource / model_type
            path.mkdir(parents=True, exist_ok=True)
            sections[resource][model_type] = {"path": str(path)}
    components = tmp_path / "models" / "flux_components"
    components.mkdir(parents=True, exist_ok=True)
    data = {
        "checkpoints": sections["checkpoints"],
        "loras": sections["loras"],
        "vaes": sections["vaes"],
        "flux": {"components_path": str(components)},
        "civitai": civitai or {},
        "api": api or {},
    }
    return AppConfig(config_file="test", parameters=Parameters(data))


@pytest.fixture
def app_config(tmp_path, monkeypatch):
    monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
    return make_config(tmp_path)
