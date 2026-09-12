"""Vercel entrypoint for the Civica FastAPI application."""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "Civica-main"
sys.path.insert(0, str(APP_DIR))

from api import app  # noqa: E402,F401

__all__ = ["app"]
