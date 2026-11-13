"""Shared test helpers."""

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def civitai_fixture(name: str):
    return json.loads((FIXTURES / "civitai" / f"{name}.json").read_text())


def flux_header_keys(name: str):
    return json.loads((FIXTURES / "flux_headers" / f"{name}.json").read_text())["keys"]
