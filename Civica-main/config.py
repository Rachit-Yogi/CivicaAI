"""Environment-backed configuration for Civica AI."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    text_provider: str = os.getenv("CIVICA_TEXT_PROVIDER", "google").lower()
    multimodal_provider: str = os.getenv("CIVICA_MULTIMODAL_PROVIDER", "google").lower()
    chat_provider: str = os.getenv("CIVICA_CHAT_PROVIDER", "google").lower()
    text_model: str = os.getenv("CIVICA_TEXT_MODEL", "gemini-3.6-flash")
    multimodal_model: str = os.getenv("CIVICA_MULTIMODAL_MODEL", "gemini-3.6-flash")
    chat_model: str = os.getenv("CIVICA_CHAT_MODEL", "gemini-3.6-flash")
    openai_reasoning_model: str = os.getenv("CIVICA_OPENAI_REASONING_MODEL", "o3")
    max_upload_mb: int = int(os.getenv("CIVICA_MAX_UPLOAD_MB", "10"))
    request_timeout_seconds: int = int(os.getenv("CIVICA_REQUEST_TIMEOUT", "15"))


settings = Settings()
