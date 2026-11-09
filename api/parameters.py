#!/usr/bin/env python
"""
Dynamic Parameters Container

Recursively converts nested dictionaries into object attributes
for convenient dot-notation access to configuration values.
"""

from typing import Any, Dict


class Parameters:
    """
    Dynamic parameter container that converts dict to object attributes.

    Allows accessing nested configuration like:
        params.checkpoints.sd15.path
    instead of:
        params['checkpoints']['sd15']['path']
    """

    def __init__(self, data: Dict[str, Any]):
        """
        Initialize Parameters from a dictionary.

        Args:
            data: Dictionary to convert to object attributes

        Raises:
            TypeError: If data is not a dictionary
        """
        if not isinstance(data, dict):
            raise TypeError(f"Parameters: expected 'dict', got {type(data).__name__}.")

        self._data = data

        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively convert nested dicts
                setattr(self, key, Parameters(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        return f"Parameters({list(self._data.keys())})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to a dictionary."""
        result = {}
        for key, value in self._data.items():
            if isinstance(value, dict):
                attr = getattr(self, key)
                result[key] = attr.to_dict() if isinstance(attr, Parameters) else value
            else:
                result[key] = value
        return result
