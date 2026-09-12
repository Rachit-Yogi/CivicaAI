"""Vercel entrypoint for the Civica FastAPI application."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "Civica-main"
APP_FILE = APP_DIR / "api.py"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

spec = importlib.util.spec_from_file_location("civica_fastapi_app", APP_FILE)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load FastAPI application from {APP_FILE}")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

app = module.app

__all__ = ["app"]
