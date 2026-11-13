"""Civitai configuration, secret masking and shared type validators."""

import pytest
from fastapi import HTTPException

from api.config import AppConfig, CIVITAI_DEFAULT_BASE_URL
from api.constants import MODEL_TYPES, RESOURCE_TYPES, validate_model_type, validate_resource_type
from api.parameters import Parameters
from metastable import ConfigurationError


def config(data):
    return AppConfig(config_file="test", parameters=Parameters(data))


@pytest.fixture(autouse=True)
def no_env_key(monkeypatch):
    monkeypatch.delenv("CIVITAI_API_KEY", raising=False)


def test_civitai_defaults_without_section():
    cfg = config({})
    assert cfg.civitai_enabled is True
    assert cfg.civitai_api_key is None
    assert cfg.civitai_base_url == CIVITAI_DEFAULT_BASE_URL
    assert cfg.civitai_nsfw is False
    assert cfg.civitai_search_cache_ttl == 60
    assert cfg.civitai_request_timeout == 30
    assert cfg.civitai_min_free_space_gb == 5
    assert cfg.job_queue_io_workers == 2


def test_civitai_settings_are_read():
    cfg = config({
        "civitai": {"enabled": False, "api_key": " abc123 ", "base_url": "http://mirror/api/v1/",
                    "nsfw": True, "search_cache_ttl": 0, "request_timeout": 5, "min_free_space_gb": 0.5},
        "api": {"job_queue": {"io_workers": 4}},
    })
    assert cfg.civitai_enabled is False
    assert cfg.civitai_api_key == "abc123"
    assert cfg.civitai_base_url == "http://mirror/api/v1"
    assert cfg.civitai_nsfw is True
    assert cfg.civitai_search_cache_ttl == 0
    assert cfg.civitai_request_timeout == 5
    assert cfg.civitai_min_free_space_gb == 0.5
    assert cfg.job_queue_io_workers == 4


def test_empty_api_key_is_none():
    assert config({"civitai": {"api_key": "  "}}).civitai_api_key is None


def test_env_api_key_wins(monkeypatch):
    monkeypatch.setenv("CIVITAI_API_KEY", "from-env")
    assert config({"civitai": {"api_key": "from-file"}}).civitai_api_key == "from-env"


@pytest.mark.parametrize("key,value", [("request_timeout", 0), ("search_cache_ttl", -1), ("min_free_space_gb", "5")])
def test_invalid_civitai_numbers_are_rejected(key, value):
    cfg = config({"civitai": {key: value}})
    with pytest.raises(ConfigurationError):
        getattr(cfg, f"civitai_{key}")


def test_api_key_is_masked_in_as_dict():
    cfg = config({"civitai": {"api_key": "secret"}, "api": {"port": 8000}})
    shown = cfg.as_dict()["config_parameters"]
    assert shown["civitai"]["api_key"] == "***"
    assert shown["api"] == {"port": 8000}
    # masking must not leak back into the live configuration
    assert cfg.civitai_api_key == "secret"


def test_empty_api_key_is_not_masked():
    assert config({"civitai": {"api_key": ""}}).as_dict()["config_parameters"]["civitai"]["api_key"] == ""


def test_type_validators_accept_known_values():
    for model_type in MODEL_TYPES:
        assert validate_model_type(model_type) == model_type
    for resource_type in RESOURCE_TYPES:
        assert validate_resource_type(resource_type) == resource_type


@pytest.mark.parametrize("validator", [validate_model_type, validate_resource_type])
def test_type_validators_reject_unknown_values(validator):
    with pytest.raises(HTTPException) as exc:
        validator("sd3")
    assert exc.value.status_code == 400
