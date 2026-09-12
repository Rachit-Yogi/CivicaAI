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
    mistral_api_key: str | None = os.getenv("MISTRAL_API_KEY")
    text_provider: str = os.getenv("CIVICA_TEXT_PROVIDER", "google").lower()
    multimodal_provider: str = os.getenv("CIVICA_MULTIMODAL_PROVIDER", "google").lower()
    chat_provider: str = os.getenv("CIVICA_CHAT_PROVIDER", "google").lower()
    text_model: str = os.getenv("CIVICA_TEXT_MODEL", "gemini-3.6-flash")
    multimodal_model: str = os.getenv("CIVICA_MULTIMODAL_MODEL", "gemini-3.6-flash")
    chat_model: str = os.getenv("CIVICA_CHAT_MODEL", "gemini-3.6-flash")
    mistral_model: str = os.getenv("CIVICA_MISTRAL_MODEL", "mistral-large-latest")
    openai_reasoning_model: str = os.getenv("CIVICA_OPENAI_REASONING_MODEL", "o3")
    fallback_provider: str | None = os.getenv("CIVICA_FALLBACK_PROVIDER", "mistral")
    fallback_model: str | None = os.getenv("CIVICA_FALLBACK_MODEL", "mistral-large-latest")
    max_upload_mb: int = int(os.getenv("CIVICA_MAX_UPLOAD_MB", "10"))
    request_timeout_seconds: int = int(os.getenv("CIVICA_REQUEST_TIMEOUT", "15"))
    llm_requests_per_minute: int = int(os.getenv("CIVICA_LLM_RPM", "8"))
    llm_max_concurrency: int = int(os.getenv("CIVICA_LLM_MAX_CONCURRENCY", "2"))
    llm_max_retries: int = int(os.getenv("CIVICA_LLM_MAX_RETRIES", "2"))
    llm_backoff_initial_seconds: float = float(os.getenv("CIVICA_LLM_BACKOFF_INITIAL", "1.0"))
    llm_backoff_max_seconds: float = float(os.getenv("CIVICA_LLM_BACKOFF_MAX", "8.0"))
    llm_cache_ttl_seconds: int = int(os.getenv("CIVICA_LLM_CACHE_TTL", "20"))
    llm_circuit_open_seconds: int = int(os.getenv("CIVICA_LLM_CIRCUIT_OPEN", "60"))


settings = Settings()
