#!/usr/bin/env python
"""
Centralized Application Configuration

Provides a single point of configuration loading and caching for the API.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

from yaml import safe_load, YAMLError

from libs.shared.parameters import Parameters
from libs.shared.utils import check_or_create_path
from libs.shared.exceptions import ConfigurationError


@dataclass
class AppConfig:
    """
    Centralized application configuration container.

    Provides access to all configuration parameters and utility methods
    for path management.
    """

    config_file: str
    parameters: Parameters

    @property
    def checkpoints_sd15_path(self) -> str:
        return self.parameters.checkpoints.sd15.path

    @property
    def checkpoints_sdxl_path(self) -> str:
        return self.parameters.checkpoints.sdxl.path

    @property
    def checkpoints_illustrious_path(self) -> str:
        return self.parameters.checkpoints.illustrious.path

    @property
    def checkpoints_flux_path(self) -> str:
        return self.parameters.checkpoints.flux.path

    @property
    def loras_sd15_path(self) -> str:
        return self.parameters.loras.sd15.path

    @property
    def loras_sdxl_path(self) -> str:
        return self.parameters.loras.sdxl.path

    @property
    def loras_illustrious_path(self) -> str:
        return self.parameters.loras.illustrious.path

    @property
    def loras_flux_path(self) -> str:
        return self.parameters.loras.flux.path

    @property
    def vae_sd15_path(self) -> str:
        return self.parameters.vaes.sd15.path

    @property
    def vae_sdxl_path(self) -> str:
        return self.parameters.vaes.sdxl.path

    @property
    def vae_illustrious_path(self) -> str:
        return self.parameters.vaes.illustrious.path

    @property
    def vae_flux_path(self) -> str:
        return self.parameters.vaes.flux.path

    @property
    def flux_components_path(self) -> str:
        """Shared diffusers-format directory holding Flux's text encoders,
        tokenizers, VAE, and scheduler (not model-checkpoint-specific)."""
        return self.parameters.flux.components_path

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
    def api_workers(self) -> int:
        return getattr(self.parameters.api, "workers", 1)

    @property
    def cors_enabled(self) -> bool:
        return getattr(self.parameters.api.cors, "enabled", True)

    @property
    def cors_allow_origins(self) -> list:
        return getattr(self.parameters.api.cors, "allow_origins", ["*"])

    def as_dict(self) -> dict:
        return {
            "config_file": self.config_file,
            "config_parameters": self.parameters.to_dict(),
        }

    def get_model_paths(self, model_type: str) -> Dict[str, str]:
        """
        Get all paths for a specific model type.

        Args:
            model_type: One of 'sd15', 'sdxl', 'illustrious', or 'flux'

        Returns:
            Dictionary with 'checkpoints', 'loras', and 'vae' paths

        Raises:
            ConfigurationError: If model_type is not a supported architecture
        """
        if model_type not in ("sd15", "sdxl", "illustrious", "flux"):
            raise ConfigurationError(
                f"Invalid model_type: {model_type}. Must be 'sd15', 'sdxl', 'illustrious', or 'flux'"
            )

        if model_type == "sd15":
            return {
                "checkpoints": self.checkpoints_sd15_path,
                "loras": self.loras_sd15_path,
                "vaes": self.vae_sd15_path,
            }
        elif model_type == "sdxl":
            return {
                "checkpoints": self.checkpoints_sdxl_path,
                "loras": self.loras_sdxl_path,
                "vaes": self.vae_sdxl_path,
            }
        elif model_type == "illustrious":
            return {
                "checkpoints": self.checkpoints_illustrious_path,
                "loras": self.loras_illustrious_path,
                "vaes": self.vae_illustrious_path,
            }
        else:  # flux
            return {
                "checkpoints": self.checkpoints_flux_path,
                "loras": self.loras_flux_path,
                "vaes": self.vae_flux_path,
            }

    def setup_paths(self) -> None:
        """Create all required directories if they don't exist."""
        paths = [
            self.checkpoints_sd15_path,
            self.checkpoints_sdxl_path,
            self.checkpoints_illustrious_path,
            self.checkpoints_flux_path,
            self.loras_sd15_path,
            self.loras_sdxl_path,
            self.loras_illustrious_path,
            self.loras_flux_path,
            self.vae_sd15_path,
            self.vae_sdxl_path,
            self.vae_illustrious_path,
            self.vae_flux_path,
            self.flux_components_path,
            self.output_images_path,
            self.output_json_path,
        ]
        for path in paths:
            check_or_create_path(path)

    def get_output_path(self, filename: str, output_type: str = "images") -> Path:
        """
        Get full output path for a file.

        Args:
            filename: The filename to save
            output_type: Either 'images' or 'json'

        Returns:
            Full Path object to the output file

        Raises:
            ConfigurationError: If output_type is not 'images' or 'json'
        """
        if output_type not in ("images", "json"):
            raise ConfigurationError(
                f"Invalid output_type: {output_type}. Must be 'images' or 'json'"
            )

        if output_type == "images":
            base_path = self.output_images_path
        else:  # json
            base_path = self.output_json_path

        return Path(base_path) / filename


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
    import os

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


def clear_config_cache() -> None:
    """Clear the configuration cache (useful for testing or reloading)."""
    get_app_config.cache_clear()


# Convenience function for common pattern
def get_model_paths(model_type: str) -> Dict[str, str]:
    """Get model paths for the given model type."""
    return get_app_config().get_model_paths(model_type)
