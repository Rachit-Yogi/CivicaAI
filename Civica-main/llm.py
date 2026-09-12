"""Resilient Google/OpenAI SDK adapters used by Civica's LangGraph workflows."""
from __future__ import annotations

import hashlib
import random
import threading
import time
from collections import deque
from typing import TypeVar

from config import settings

T = TypeVar("T")

_GOOGLE_LEGACY_MODEL_MAP = {
    "gemini-2.5-flash": "gemini-3.6-flash",
    "models/gemini-2.5-flash": "gemini-3.6-flash",
}

_request_lock = threading.Lock()
_request_times: dict[str, deque[float]] = {}
_concurrency: dict[str, threading.BoundedSemaphore] = {}
_cache_lock = threading.Lock()
_response_cache: dict[str, tuple[float, object]] = {}
_circuit_lock = threading.Lock()
_circuit_open_until: dict[str, float] = {}


class LLMRateLimitError(RuntimeError):
    """Raised when the configured provider cannot currently accept requests."""


class LLMQuotaError(LLMRateLimitError):
    """Raised when provider quota is exhausted and retries are not useful."""


def _google_client():
    from google import genai
    from google.genai import types

    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for the configured Google model.")

    # Application code owns retries, so the SDK performs one HTTP attempt per
    # application attempt. This avoids multiplying retry traffic against quota.
    retry_options = types.HttpRetryOptions(
        attempts=1,
        initial_delay=1.0,
        max_delay=1.0,
        exp_base=2.0,
        jitter=0.0,
        http_status_codes=[408, 429, 500, 502, 503, 504],
    )
    return genai.Client(
        api_key=settings.google_api_key,
        http_options=types.HttpOptions(
            timeout=max(1, settings.request_timeout_seconds) * 1000,
            retry_options=retry_options,
        ),
    )


def _openai_client():
    from openai import OpenAI

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the configured OpenAI model.")
    return OpenAI(api_key=settings.openai_api_key, timeout=settings.request_timeout_seconds)


def _normalize_model(provider: str, model: str) -> str:
    if provider == "google":
        return _GOOGLE_LEGACY_MODEL_MAP.get(model.strip(), model.strip())
    return model.strip()


def _model_config(task: str) -> tuple[str, str]:
    if task == "multimodal":
        provider, model = settings.multimodal_provider, settings.multimodal_model
    elif task == "chat":
        provider, model = settings.chat_provider, settings.chat_model
    elif task == "reasoning":
        provider, model = "openai", settings.openai_reasoning_model
    else:
        provider, model = settings.text_provider, settings.text_model
    return provider, _normalize_model(provider, model)


def _fallback_config(task: str, primary_provider: str, primary_model: str, grounded: bool) -> tuple[str, str] | None:
    # Grounded scheme/fraud research must not silently lose live evidence.
    # Multimodal fallback is disabled because the current OpenAI path is text-only.
    if grounded or task == "multimodal":
        return None
    provider = (settings.fallback_provider or "").strip().lower()
    model = (settings.fallback_model or "").strip()
    if not provider or not model or provider == primary_provider:
        return None
    if provider == "google" and not settings.google_api_key:
        return None
    if provider == "openai" and not settings.openai_api_key:
        return None
    return provider, _normalize_model(provider, model)


def _google_config(*, types, temperature: float, schema=None, grounded: bool = False):
    kwargs = {"temperature": temperature}
    if schema is not None:
        kwargs.update({"response_mime_type": "application/json", "response_schema": schema})
    if grounded:
        kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    return types.GenerateContentConfig(**kwargs)


def _slot_for(provider: str) -> threading.BoundedSemaphore:
    key = provider.lower()
    with _request_lock:
        if key not in _concurrency:
            _concurrency[key] = threading.BoundedSemaphore(max(1, settings.llm_max_concurrency))
        return _concurrency[key]


def _acquire_request_slot(provider: str) -> threading.BoundedSemaphore:
    """Apply a sliding-window RPM limit before hitting the provider."""
    semaphore = _slot_for(provider)
    semaphore.acquire()
    try:
        rpm = max(1, settings.llm_requests_per_minute)
        window = 60.0
        while True:
            now = time.monotonic()
            with _request_lock:
                timestamps = _request_times.setdefault(provider.lower(), deque())
                while timestamps and now - timestamps[0] >= window:
                    timestamps.popleft()
                if len(timestamps) < rpm:
                    timestamps.append(now)
                    return semaphore
                sleep_for = max(0.05, window - (now - timestamps[0]))
            time.sleep(sleep_for)
    except Exception:
        semaphore.release()
        raise


def _release_request_slot(semaphore: threading.BoundedSemaphore) -> None:
    semaphore.release()


def _cache_key(*, task: str, provider: str, model: str, prompt: str, grounded: bool, image_bytes: bytes | None) -> str:
    payload = bytearray()
    payload.extend(task.encode())
    payload.extend(b"\0")
    payload.extend(provider.encode())
    payload.extend(b"\0")
    payload.extend(model.encode())
    payload.extend(b"\0")
    payload.extend(prompt.encode())
    payload.extend(b"\0")
    payload.extend(str(grounded).encode())
    if image_bytes:
        payload.extend(hashlib.sha256(image_bytes).digest())
    return hashlib.sha256(payload).hexdigest()


def _cache_get(key: str):
    if settings.llm_cache_ttl_seconds <= 0:
        return None
    now = time.monotonic()
    with _cache_lock:
        item = _response_cache.get(key)
        if not item:
            return None
        created, value = item
        if now - created > settings.llm_cache_ttl_seconds:
            _response_cache.pop(key, None)
            return None
        return value


def _cache_put(key: str, value: object) -> None:
    if settings.llm_cache_ttl_seconds <= 0:
        return
    with _cache_lock:
        _response_cache[key] = (time.monotonic(), value)
        if len(_response_cache) > 256:
            oldest = min(_response_cache.items(), key=lambda item: item[1][0])[0]
            _response_cache.pop(oldest, None)


def _error_text(exc: Exception) -> str:
    return str(getattr(exc, "message", None) or str(exc)).lower()


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    text = _error_text(exc)
    return code in {408, 429, 500, 502, 503, 504} or any(
        marker in text
        for marker in ("429", "resource_exhausted", "rate limit", "too many requests", "unavailable", "timeout")
    )


def _is_daily_quota(exc: Exception) -> bool:
    text = _error_text(exc)
    return any(
        marker in text
        for marker in (
            "daily quota",
            "quota_exceeded",
            "quota has been exhausted",
            "exceeded your current quota",
            "quota will reset",
        )
    )


def _circuit_key(provider: str, model: str) -> str:
    return f"{provider.lower()}:{model}"


def _check_circuit(provider: str, model: str) -> None:
    key = _circuit_key(provider, model)
    now = time.monotonic()
    with _circuit_lock:
        until = _circuit_open_until.get(key, 0.0)
        if until > now:
            raise LLMQuotaError(
                "The AI service quota is temporarily exhausted. Please try again later or switch to a configured fallback provider."
            )
        if until:
            _circuit_open_until.pop(key, None)


def _open_circuit(provider: str, model: str) -> None:
    with _circuit_lock:
        _circuit_open_until[_circuit_key(provider, model)] = time.monotonic() + max(
            30, settings.llm_circuit_open_seconds
        )


def _backoff(attempt: int) -> None:
    base = min(
        settings.llm_backoff_max_seconds,
        settings.llm_backoff_initial_seconds * (2 ** attempt),
    )
    jitter = random.uniform(0, min(1.0, base * 0.25))
    time.sleep(max(0.05, base + jitter))


def _provider_call(provider: str, model: str, operation):
    """Run a provider call with burst protection and bounded retry/backoff."""
    _check_circuit(provider, model)
    semaphore = _acquire_request_slot(provider)
    try:
        attempts = max(1, settings.llm_max_retries + 1)
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return operation()
            except Exception as exc:  # SDK/provider-specific exception types vary by version.
                last_exc = exc
                if _is_daily_quota(exc):
                    _open_circuit(provider, model)
                    raise LLMQuotaError(
                        "The configured AI provider has exhausted its quota. Please check billing/quota or use a configured fallback provider."
                    ) from exc
                if not _is_retryable(exc) or attempt >= attempts - 1:
                    break
                _backoff(attempt)
        assert last_exc is not None
        if getattr(last_exc, "code", None) == 429 or "429" in _error_text(last_exc):
            raise LLMRateLimitError(
                "The AI provider is rate-limited right now. Please retry shortly."
            ) from last_exc
        raise last_exc
    finally:
        _release_request_slot(semaphore)


def _google_structured(*, model: str, schema: type[T], contents: list[object], temperature: float, grounded: bool) -> T:
    from google.genai import types

    with _google_client() as client:
        response = _provider_call(
            "google",
            model,
            lambda: client.models.generate_content(
                model=model,
                contents=contents,
                config=_google_config(types=types, temperature=temperature, schema=schema, grounded=grounded),
            ),
        )
    return schema.model_validate_json(response.text)


def generate_structured(*, task: str, prompt: str, schema: type[T], image_bytes: bytes | None = None,
                        image_mime_type: str | None = None, temperature: float = 0.2,
                        grounded: bool = False) -> T:
    provider, model = _model_config(task)
    key = _cache_key(task=task, provider=provider, model=model, prompt=prompt, grounded=grounded, image_bytes=image_bytes)
    cached = _cache_get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    contents: list[object] = [prompt]
    if image_bytes is not None:
        if not image_mime_type:
            raise ValueError("image_mime_type is required when image_bytes is supplied.")
        from google.genai import types
        contents = [types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type), prompt]

    try:
        if provider == "google":
            result = _google_structured(
                model=model,
                schema=schema,
                contents=contents,
                temperature=temperature,
                grounded=grounded,
            )
        elif provider == "openai":
            client = _openai_client()
            try:
                response = _provider_call(
                    "openai",
                    model,
                    lambda: client.responses.parse(model=model, input=prompt, text_format=schema),
                )
            finally:
                client.close()
            if response.output_parsed is None:
                raise RuntimeError("OpenAI returned no structured output.")
            result = response.output_parsed
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    except (LLMRateLimitError, LLMQuotaError) as primary_error:
        fallback = _fallback_config(task, provider, model, grounded)
        if not fallback:
            raise primary_error
        fallback_provider, fallback_model = fallback
        if fallback_provider == "openai":
            client = _openai_client()
            try:
                response = _provider_call(
                    "openai",
                    fallback_model,
                    lambda: client.responses.parse(model=fallback_model, input=prompt, text_format=schema),
                )
            finally:
                client.close()
            if response.output_parsed is None:
                raise primary_error
            result = response.output_parsed
        else:
            result = _google_structured(
                model=fallback_model,
                schema=schema,
                contents=contents,
                temperature=temperature,
                grounded=False,
            )

    _cache_put(key, result)
    return result


def _google_text(*, model: str, prompt: str, temperature: float, grounded: bool) -> str:
    from google.genai import types

    with _google_client() as client:
        response = _provider_call(
            "google",
            model,
            lambda: client.models.generate_content(
                model=model,
                contents=prompt,
                config=_google_config(types=types, temperature=temperature, grounded=grounded),
            ),
        )
    return response.text or ""


def generate_text(*, task: str, prompt: str, temperature: float = 0.4, grounded: bool = False) -> str:
    provider, model = _model_config(task)
    key = _cache_key(task=task, provider=provider, model=model, prompt=prompt, grounded=grounded, image_bytes=None)
    cached = _cache_get(key)
    if cached is not None:
        return str(cached)

    try:
        if provider == "google":
            result = _google_text(model=model, prompt=prompt, temperature=temperature, grounded=grounded)
        elif provider == "openai":
            client = _openai_client()
            try:
                response = _provider_call("openai", model, lambda: client.responses.create(model=model, input=prompt))
            finally:
                client.close()
            result = response.output_text or ""
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    except (LLMRateLimitError, LLMQuotaError) as primary_error:
        fallback = _fallback_config(task, provider, model, grounded)
        if not fallback:
            raise primary_error
        fallback_provider, fallback_model = fallback
        if fallback_provider == "openai":
            client = _openai_client()
            try:
                response = _provider_call("openai", fallback_model, lambda: client.responses.create(model=fallback_model, input=prompt))
            finally:
                client.close()
            result = response.output_text or ""
        else:
            result = _google_text(model=fallback_model, prompt=prompt, temperature=temperature, grounded=False)

    _cache_put(key, result)
    return result


def model_info(task: str) -> dict[str, str]:
    provider, model = _model_config(task)
    return {"provider": provider, "model": model}
