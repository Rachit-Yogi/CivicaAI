"""Resilient Google/OpenAI/Mistral LLM adapters with grounded web-search fallback."""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections import deque
from typing import Any, TypeVar

import requests

from config import settings

T = TypeVar("T")

_GOOGLE_LEGACY_MODEL_MAP = {
    "gemini-2.5-flash": "gemini-3.6-flash",
    "models/gemini-2.5-flash": "gemini-3.6-flash",
}
_MISTRAL_API_URL = "https://api.mistral.ai/v1/conversations"

_request_lock = threading.Lock()
_request_times: dict[str, deque[float]] = {}
_concurrency: dict[str, threading.BoundedSemaphore] = {}
_cache_lock = threading.Lock()
_response_cache: dict[str, tuple[float, object]] = {}
_circuit_lock = threading.Lock()
_circuit_open_until: dict[str, float] = {}


class LLMRateLimitError(RuntimeError):
    """Raised when a provider is temporarily unavailable or rate-limited."""


class LLMQuotaError(LLMRateLimitError):
    """Raised when a provider quota is exhausted."""


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


def _fallback_config(task: str, primary_provider: str, grounded: bool) -> tuple[str, str] | None:
    provider = (settings.fallback_provider or "").strip().lower()
    model = (settings.fallback_model or "").strip()
    if not provider or provider == primary_provider:
        return None
    if provider == "google" and not settings.google_api_key:
        return None
    if provider == "openai" and not settings.openai_api_key:
        return None
    if provider == "mistral" and not settings.mistral_api_key:
        return None
    if provider == "mistral" and (not model or model.startswith("gemini-")):
        model = settings.mistral_model
    if not model:
        return None
    # Mistral web_search is available through Conversations/Agents, not Chat Completions.
    # Grounded fallbacks therefore use a dedicated Conversations path below.
    return provider, _normalize_model(provider, model)


def _google_config(*, types, temperature: float, schema=None, grounded: bool = False):
    kwargs: dict[str, Any] = {"temperature": temperature}
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
        while True:
            now = time.monotonic()
            with _request_lock:
                timestamps = _request_times.setdefault(provider.lower(), deque())
                while timestamps and now - timestamps[0] >= 60:
                    timestamps.popleft()
                if len(timestamps) < rpm:
                    timestamps.append(now)
                    return semaphore
                sleep_for = max(0.05, 60 - (now - timestamps[0]))
            time.sleep(sleep_for)
    except Exception:
        semaphore.release()
        raise


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
    with _cache_lock:
        item = _response_cache.get(key)
        if not item:
            return None
        created, value = item
        if time.monotonic() - created > settings.llm_cache_ttl_seconds:
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


def _error_code(exc: Exception) -> int | None:
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
            "408", "429", "resource_exhausted", "rate limit", "too many requests",
            "temporarily unavailable", "service unavailable", "gateway timeout",
            "timeout", "connection reset", "connection refused",
        )
    )


def _is_quota_error(exc: Exception) -> bool:
    text = _error_text(exc)
    return any(marker in text for marker in (
        "daily quota", "quota_exceeded", "quota has been exhausted",
        "exceeded your current quota", "quota will reset", "insufficient quota",
    ))


def _circuit_key(provider: str, model: str) -> str:
    return f"{provider.lower()}:{model}"


def _check_circuit(provider: str, model: str) -> None:
    now = time.monotonic()
    key = _circuit_key(provider, model)
    with _circuit_lock:
        until = _circuit_open_until.get(key, 0.0)
        if until > now:
            raise LLMQuotaError("The AI service is temporarily unavailable; using the configured fallback provider.")
        if until:
            _circuit_open_until.pop(key, None)


def _open_circuit(provider: str, model: str) -> None:
    with _circuit_lock:
        _circuit_open_until[_circuit_key(provider, model)] = time.monotonic() + max(30, settings.llm_circuit_open_seconds)


def _backoff(attempt: int) -> None:
    base = min(settings.llm_backoff_max_seconds, settings.llm_backoff_initial_seconds * (2 ** attempt))
    time.sleep(max(0.05, base + random.uniform(0, min(1.0, base * 0.25))))


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
                if _is_quota_error(exc):
                    _open_circuit(provider, model)
                    raise LLMQuotaError("The configured AI provider has exhausted its quota.") from exc
                if not _is_retryable(exc) or attempt >= attempts - 1:
                    break
                _backoff(attempt)
        assert last_exc is not None
        if _is_retryable(last_exc):
            raise LLMRateLimitError("The AI provider is temporarily unavailable or rate-limited.") from last_exc
        raise last_exc
    finally:
        semaphore.release()


def _mistral_http(method: str, payload: dict[str, Any]):
    if not settings.mistral_api_key:
        raise RuntimeError("MISTRAL_API_KEY is required for Mistral.")
    response = requests.request(
        method,
        _MISTRAL_API_URL,
        headers={
            "Authorization": f"Bearer {settings.mistral_api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=settings.request_timeout_seconds,
    )
    if response.status_code >= 400:
        message = response.text[:2000]
        error = RuntimeError(f"Mistral HTTP {response.status_code}: {message}")
        setattr(error, "status_code", response.status_code)
        raise error
    return response.json()


def _mistral_output_text(response: dict[str, Any]) -> str:
    chunks: list[str] = []
    for output in response.get("outputs", []) or []:
        if output.get("type") != "message.output":
            continue
        content = output.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict):
                    if isinstance(chunk.get("text"), str):
                        chunks.append(chunk["text"])
                elif getattr(chunk, "text", None):
                    chunks.append(str(chunk.text))
    return "".join(chunks).strip()


def _mistral_text(*, model: str, prompt: str, temperature: float, grounded: bool = False) -> str:
    def call():
        if grounded:
            response = _mistral_http("POST", {
                "model": model,
                "inputs": [{"role": "user", "content": prompt}],
                "tools": [{"type": "web_search"}],
                "completion_args": {"temperature": temperature},
            })
            return _mistral_output_text(response)
        # Non-grounded Mistral uses the stable Chat Completions endpoint.
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.mistral_api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature},
            timeout=settings.request_timeout_seconds,
        )
        if response.status_code >= 400:
            error = RuntimeError(f"Mistral HTTP {response.status_code}: {response.text[:2000]}")
            setattr(error, "status_code", response.status_code)
            raise error
        data = response.json()
        return data["choices"][0]["message"].get("content", "")
    return _provider_call("mistral", model, call)


def _mistral_structured(*, model: str, prompt: str, schema: type[T], temperature: float, grounded: bool = False) -> T:
    instruction = (
        f"Return ONLY valid JSON matching this schema exactly. Do not use markdown. Schema: "
        f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}\n\nUser request:\n{prompt}"
    )
    raw = _mistral_text(model=model, prompt=instruction, temperature=temperature, grounded=grounded)
    try:
        return schema.model_validate_json(raw)
    except Exception:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise
        return schema.model_validate_json(raw[start:end + 1])


def _google_structured(*, model: str, schema: type[T], contents: list[object], temperature: float, grounded: bool) -> T:
    from google.genai import types
    with _google_client() as client:
        response = _provider_call(
            "google", model,
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
        response = _provider_call("openai", model, lambda: client.responses.parse(model=model, input=prompt, text_format=schema))
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


def _fallback_result(*, task: str, provider: str, prompt: str, schema=None, temperature: float, grounded: bool):
    fallback = _fallback_config(task, provider, grounded)
    if not fallback:
        raise LLMRateLimitError("No configured fallback provider is available.")
    fallback_provider, fallback_model = fallback
    if schema is not None:
        if fallback_provider == "mistral":
            return _mistral_structured(model=fallback_model, prompt=prompt, schema=schema, temperature=temperature, grounded=grounded)
        if fallback_provider == "openai":
            return _openai_structured(model=fallback_model, prompt=prompt, schema=schema)
        return _google_structured(model=fallback_model, schema=schema, contents=[prompt], temperature=temperature, grounded=grounded)
    if fallback_provider == "mistral":
        return _mistral_text(model=fallback_model, prompt=prompt, temperature=temperature, grounded=grounded)
    if fallback_provider == "openai":
        return _openai_text(model=fallback_model, prompt=prompt)
    from google.genai import types
    with _google_client() as client:
        response = _provider_call("google", fallback_model, lambda: client.models.generate_content(model=fallback_model, contents=prompt, config=_google_config(types=types, temperature=temperature, grounded=grounded)))
    return response.text or ""


def generate_structured(*, task: str, prompt: str, schema: type[T], image_bytes: bytes | None = None, image_mime_type: str | None = None, temperature: float = 0.2, grounded: bool = False) -> T:
    provider, model = _model_config(task)
    key = _cache_key(task=task, provider=provider, model=model, prompt=prompt, grounded=grounded, image_bytes=image_bytes)
    cached = _cache_get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    if image_bytes is not None and provider != "google":
        raise ValueError(f"Provider '{provider}' does not support this multimodal path.")
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
            result = _openai_structured(model=model, prompt=prompt, schema=schema)
        elif provider == "mistral":
            result = _mistral_structured(model=model, prompt=prompt, schema=schema, temperature=temperature, grounded=grounded)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    except (LLMRateLimitError, LLMQuotaError):
        result = _fallback_result(task=task, provider=provider, prompt=prompt, schema=schema, temperature=temperature, grounded=grounded)
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
                response = _provider_call("google", model, lambda: client.models.generate_content(model=model, contents=prompt, config=_google_config(types=types, temperature=temperature, grounded=grounded)))
            result = response.text or ""
        elif provider == "openai":
            result = _openai_text(model=model, prompt=prompt)
        elif provider == "mistral":
            result = _mistral_text(model=model, prompt=prompt, temperature=temperature, grounded=grounded)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    except (LLMRateLimitError, LLMQuotaError):
        result = _fallback_result(task=task, provider=provider, prompt=prompt, temperature=temperature, grounded=grounded)
    _cache_put(key, result)
    return result


def model_info(task: str) -> dict[str, str]:
    provider, model = _model_config(task)
    return {"provider": provider, "model": model}
