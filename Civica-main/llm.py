"""Resilient Google/OpenAI/Mistral SDK adapters used by Civica's LangGraph workflows."""
from __future__ import annotations

import hashlib
import json
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


def _mistral_client():
    from mistralai.client import Mistral

    if not settings.mistral_api_key:
        raise RuntimeError("MISTRAL_API_KEY is required for the configured Mistral model.")
    return Mistral(api_key=settings.mistral_api_key)


def _normalize_model(provider: str, model: str) -> str:
    provider = provider.lower()
    model = model.strip()
    if provider == "google":
        return _GOOGLE_LEGACY_MODEL_MAP.get(model, model)
    return model


def _model_config(task: str) -> tuple[str, str]:
    if task == "multimodal":
        provider, model = settings.multimodal_provider, settings.multimodal_model
    elif task == "chat":
        provider, model = settings.chat_provider, settings.chat_model
    elif task == "reasoning":
        provider, model = "openai", settings.openai_reasoning_model
    else:
        provider, model = settings.text_provider, settings.text_model

    provider = provider.lower()
    if provider == "mistral" and model.startswith("gemini-"):
        model = settings.mistral_model
    return provider, _normalize_model(provider, model)


def _fallback_config(task: str, primary_provider: str, primary_model: str, grounded: bool) -> tuple[str, str] | None:
    # Grounded scheme/fraud research must not silently lose live evidence.
    # Multimodal fallback is disabled because the current Mistral/OpenAI fallbacks
    # in this service are text-only.
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
    if provider == "mistral" and not settings.mistral_api_key:
        return None
    if provider == "mistral" and model.startswith("gemini-"):
        model = settings.mistral_model
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
    for value in (task, provider, model, prompt, str(grounded)):
        payload.extend(value.encode())
        payload.extend(b"\0")
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


def _error_code(exc: Exception):
    for name in ("code", "status_code", "status"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _is_retryable(exc: Exception) -> bool:
    code = _error_code(exc)
    text = _error_text(exc)
    return code in {408, 409, 429, 500, 502, 503, 504} or any(
        marker in text
        for marker in (
            "429",
            "resource_exhausted",
            "rate limit",
            "too many requests",
            "temporarily unavailable",
            "service unavailable",
            "gateway timeout",
            "timeout",
            "connection reset",
        )
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
            "insufficient quota",
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
                "The AI service is temporarily unavailable. Please try again later or use the configured fallback provider."
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
        settings.llm_backoff_initial_seconds * (2**attempt),
    )
    jitter = random.uniform(0, min(1.0, base * 0.25))
    time.sleep(max(0.05, base + jitter))


def _provider_call(provider: str, model: str, operation):
    _check_circuit(provider, model)
    semaphore = _acquire_request_slot(provider)
    try:
        attempts = max(1, settings.llm_max_retries + 1)
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return operation()
            except Exception as exc:
                last_exc = exc
                if _is_daily_quota(exc):
                    _open_circuit(provider, model)
                    raise LLMQuotaError(
                        "The configured AI provider has exhausted its quota. Please use the configured fallback provider or check provider billing/quota."
                    ) from exc
                if not _is_retryable(exc) or attempt >= attempts - 1:
                    break
                _backoff(attempt)
        assert last_exc is not None
        if _error_code(last_exc) == 429 or "429" in _error_text(last_exc):
            raise LLMRateLimitError("The AI provider is rate-limited right now. Please retry shortly.") from last_exc
        raise last_exc
    finally:
        _release_request_slot(semaphore)


def _mistral_content(response) -> str:
    content = response.choices[0].message.content if response.choices else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(getattr(part, "text", ""))
            for part in content
        )
    return str(content or "")


def _mistral_text(*, model: str, prompt: str, temperature: float) -> str:
    with _mistral_client() as client:
        response = _provider_call(
            "mistral",
            model,
            lambda: client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            ),
        )
    return _mistral_content(response)


def _mistral_structured(*, model: str, prompt: str, schema: type[T], temperature: float) -> T:
    with _mistral_client() as client:
        def call():
            response = client.chat.parse(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Return only a JSON object matching the requested response schema.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=schema,
                temperature=temperature,
            )
            parsed = getattr(response.choices[0].message, "parsed", None) if response.choices else None
            if parsed is not None:
                return parsed
            raw = _mistral_content(response)
            return schema.model_validate_json(raw)

        return _provider_call("mistral", model, call)


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


def _openai_structured(*, model: str, prompt: str, schema: type[T]) -> T:
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
    return response.output_parsed


def _openai_text(*, model: str, prompt: str) -> str:
    client = _openai_client()
    try:
        response = _provider_call("openai", model, lambda: client.responses.create(model=model, input=prompt))
    finally:
        client.close()
    return response.output_text or ""


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
            result = _google_structured(model=model, schema=schema, contents=contents, temperature=temperature, grounded=grounded)
        elif provider == "openai":
            if image_bytes:
                raise ValueError("OpenAI structured multimodal requests are not supported by this workflow.")
            result = _openai_structured(model=model, prompt=prompt, schema=schema)
        elif provider == "mistral":
            if image_bytes:
                raise ValueError("Mistral structured multimodal requests are not enabled by this workflow.")
            result = _mistral_structured(model=model, prompt=prompt, schema=schema, temperature=temperature)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    except (LLMRateLimitError, LLMQuotaError) as primary_error:
        fallback = _fallback_config(task, provider, model, grounded)
        if not fallback:
            raise primary_error
        fallback_provider, fallback_model = fallback
        if fallback_provider == "mistral":
            result = _mistral_structured(model=fallback_model, prompt=prompt, schema=schema, temperature=temperature)
        elif fallback_provider == "openai":
            result = _openai_structured(model=fallback_model, prompt=prompt, schema=schema)
        else:
            result = _google_structured(model=fallback_model, schema=schema, contents=contents, temperature=temperature, grounded=False)

    _cache_put(key, result)
    return result


def generate_text(*, task: str, prompt: str, temperature: float = 0.4, grounded: bool = False) -> str:
    provider, model = _model_config(task)
    key = _cache_key(task=task, provider=provider, model=model, prompt=prompt, grounded=grounded, image_bytes=None)
    cached = _cache_get(key)
    if cached is not None:
        return str(cached)

    try:
        if provider == "google":
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
            result = response.text or ""
        elif provider == "openai":
            result = _openai_text(model=model, prompt=prompt)
        elif provider == "mistral":
            result = _mistral_text(model=model, prompt=prompt, temperature=temperature)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    except (LLMRateLimitError, LLMQuotaError) as primary_error:
        fallback = _fallback_config(task, provider, model, grounded)
        if not fallback:
            raise primary_error
        fallback_provider, fallback_model = fallback
        if fallback_provider == "mistral":
            result = _mistral_text(model=fallback_model, prompt=prompt, temperature=temperature)
        elif fallback_provider == "openai":
            result = _openai_text(model=fallback_model, prompt=prompt)
        else:
            from google.genai import types
            with _google_client() as client:
                response = _provider_call(
                    "google",
                    fallback_model,
                    lambda: client.models.generate_content(
                        model=fallback_model,
                        contents=prompt,
                        config=_google_config(types=types, temperature=temperature, grounded=False),
                    ),
                )
            result = response.text or ""

    _cache_put(key, result)
    return result


def model_info(task: str) -> dict[str, str]:
    provider, model = _model_config(task)
    return {"provider": provider, "model": model}
