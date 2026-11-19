#!/usr/bin/env python
"""
Centralized Application Configuration

Provides a single point of configuration loading and caching for the API.
"""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List, Optional

from yaml import safe_load, YAMLError

from metastable import ConfigurationError

from api.constants import (
    CIVITAI_DEFAULT_BASE_URL,
    MODEL_TYPES,
    RESOURCE_TYPES,
    TRANSFORMER_MODEL_TYPES,
    ZIMAGE_VARIANTS,
)
from api.parameters import Parameters

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """
    Centralized application configuration container.

    Provides access to all configuration parameters and utility methods
    for path management.
    """

    config_file: str
    parameters: Parameters

    def get_model_paths(self, model_type: str) -> Dict[str, str]:
        """
        Get all paths for a specific model type.

        Args:
            model_type: One of MODEL_TYPES

        Returns:
            Dictionary with 'checkpoints', 'loras', and 'vaes' paths

        Raises:
            ConfigurationError: If model_type is not a supported architecture
                or its directories are not configured
        """
        if model_type not in MODEL_TYPES:
            raise ConfigurationError(
                f"Invalid model_type: {model_type}. Must be one of {', '.join(MODEL_TYPES)}"
            )

        paths = {}
        for resource_type in RESOURCE_TYPES:
            section = getattr(getattr(self.parameters, resource_type, None), model_type, None)
            path = getattr(section, "path", None)
            if not path:
                raise ConfigurationError(
                    f"{model_type} is not configured: set {resource_type}.{model_type}.path "
                    f"in {self.config_file}"
                )
            paths[resource_type] = path
        return paths

    def is_configured(self, model_type: str) -> bool:
        """True if every resource directory of model_type is configured."""
        try:
            self.get_model_paths(model_type)
        except ConfigurationError:
            return False
        return True

    def components_path(self, model_type: str, variant: Optional[str] = None) -> str:
        """
        Shared diffusers-format directory holding the text encoder(s),
        tokenizer(s), VAE and scheduler of a transformer-only architecture
        (not model-checkpoint-specific).

        Args:
            model_type: One of TRANSFORMER_MODEL_TYPES
            variant: Z-Image only, one of ZIMAGE_VARIANTS

        Raises:
            ConfigurationError: If model_type has no components directory, or
                it is not configured
        """
        if model_type == "flux":
            key = "flux.components_path"
            path = getattr(getattr(self.parameters, "flux", None), "components_path", None)
        elif model_type == "zimage":
            if variant not in ZIMAGE_VARIANTS:
                raise ConfigurationError(
                    f"Invalid Z-Image variant: {variant}. Must be one of {', '.join(ZIMAGE_VARIANTS)}"
                )
            key = f"zimage.components.{variant}"
            components = getattr(getattr(self.parameters, "zimage", None), "components", None)
            path = getattr(components, variant, None)
        else:
            raise ConfigurationError(f"{model_type} has no shared components directory")

        if not path:
            raise ConfigurationError(f"{key} is not set in {self.config_file}")
        return path

    def _configured_components_paths(self) -> List[str]:
        variants = {"flux": (None,), "zimage": ZIMAGE_VARIANTS}
        paths = []
        for model_type in TRANSFORMER_MODEL_TYPES:
            for variant in variants[model_type]:
                try:
                    paths.append(self.components_path(model_type, variant))
                except ConfigurationError:
                    pass
        return paths

    @property
    def output_images_path(self) -> str:
        return self.parameters.storage.output_images

    @property
    def persist_to_backend(self) -> bool:
        return self.parameters.storage.persist_to_backend

    @property
    def output_json_path(self) -> str:
        return self.parameters.storage.output_json

    @property
    def api_host(self) -> str:
        return getattr(self.parameters.api, "host", "0.0.0.0")

    @property
    def api_port(self) -> int:
        return getattr(self.parameters.api, "port", 8000)

    @property
    def cors_enabled(self) -> bool:
        return getattr(self.parameters.api.cors, "enabled", True)

    @property
    def cors_allow_origins(self) -> list:
        return getattr(self.parameters.api.cors, "allow_origins", ["*"])

    def _api_setting(self, section: str, key: str, default: int) -> int:
        """Read a positive integer from parameters.api.<section>.<key>."""
        subtree = getattr(getattr(self.parameters, "api", None), section, None)
        value = getattr(subtree, key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigurationError(
                f"api.{section}.{key} must be a positive integer, got {value!r}"
            )
        return value

    @property
    def job_queue_num_workers(self) -> int:
        """Number of concurrent GPU job workers (generation and merge)."""
        return self._api_setting("job_queue", "num_workers", 1)

    @property
    def job_queue_result_ttl(self) -> int:
        """Seconds a finished job is kept before cleanup."""
        return self._api_setting("job_queue", "result_ttl", 3600)

    @property
    def model_cache_max_models(self) -> int:
        """Maximum number of pipelines kept loaded in the model cache."""
        return self._api_setting("model_cache", "max_models", 2)

    @property
    def job_queue_io_workers(self) -> int:
        """Number of concurrent I/O job workers (Civitai downloads)."""
        return self._api_setting("job_queue", "io_workers", 2)

    def _civitai_setting(self, key: str, default: Any) -> Any:
        """Read parameters.civitai.<key>; the whole section is optional."""
        return getattr(getattr(self.parameters, "civitai", None), key, default)

    def _civitai_number(self, key: str, default: float, minimum: float) -> float:
        value = self._civitai_setting(key, default)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < minimum:
            raise ConfigurationError(f"civitai.{key} must be a number >= {minimum}, got {value!r}")
        return value

    @property
    def civitai_enabled(self) -> bool:
        return bool(self._civitai_setting("enabled", True))

    @property
    def civitai_api_key(self) -> Optional[str]:
        """Civitai API key; the CIVITAI_API_KEY environment variable wins."""
        key = os.environ.get("CIVITAI_API_KEY") or self._civitai_setting("api_key", None)
        key = str(key).strip() if key else ""
        return key or None

    @property
    def civitai_base_url(self) -> str:
        return str(self._civitai_setting("base_url", CIVITAI_DEFAULT_BASE_URL)).rstrip("/")

    @property
    def civitai_nsfw(self) -> bool:
        """Default value of the NSFW toggle in Civitai searches."""
        return bool(self._civitai_setting("nsfw", False))

    @property
    def civitai_search_cache_ttl(self) -> float:
        return self._civitai_number("search_cache_ttl", 60, 0)

    @property
    def civitai_request_timeout(self) -> float:
        return self._civitai_number("request_timeout", 30, 1)

    @property
    def civitai_min_free_space_gb(self) -> float:
        return self._civitai_number("min_free_space_gb", 5, 0)

    @property
    def device(self) -> Optional[Parameters]:
        """
        The optional 'device' configuration subtree (compute backend, dtype,
        ROCm overrides), or None when parameters.yaml has no device section.
        Passed to metastable.configure_device() at startup.
        """
        device = getattr(self.parameters, "device", None)
        return device if isinstance(device, Parameters) else None

    def as_dict(self) -> dict:
        """Configuration for display; secrets are masked."""
        parameters = self.parameters.to_dict()
        civitai = parameters.get("civitai")
        if isinstance(civitai, dict) and civitai.get("api_key"):
            civitai["api_key"] = "***"
        return {
            "config_file": self.config_file,
            "config_parameters": parameters,
        }

    def setup_paths(self) -> None:
        """Create all configured directories if they don't exist."""
        paths = [self.output_images_path, self.output_json_path]
        for model_type in MODEL_TYPES:
            if self.is_configured(model_type):
                paths.extend(self.get_model_paths(model_type).values())
            else:
                logger.warning(f"{model_type} models are not configured in {self.config_file}")
        paths.extend(self._configured_components_paths())
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)


def _load_yaml_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_file, "r") as f:
            return safe_load(f)
    except YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML config: {e}")
    except FileNotFoundError:
        raise ConfigurationError(f"Config file not found: {config_file}")


@lru_cache(maxsize=1)
def get_app_config(params_file: str = "parameters.yaml") -> AppConfig:
    """
    Get the application configuration (cached).

    This function is cached so configuration is only loaded once per session.

    Args:
        params_file: Path to the parameters.yaml file

    Returns:
        AppConfig instance with all configuration loaded
    """
    # Get config file path from environment or use default
    # if env variable is specified, the it overrides the filename provided
    config_file = os.environ.get("CONFIG_FILE", params_file)

    # Load YAML configuration
    config_dict = _load_yaml_config(config_file)
    parameters = Parameters(config_dict)

    # Create and return config object
    config = AppConfig(
        config_file=config_file,
        parameters=parameters,
    )

    # Ensure all paths exist
    config.setup_paths()

    return config
