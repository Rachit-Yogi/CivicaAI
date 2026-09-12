"""Direct model adapters used by Civica's LangGraph workflows.

Provider SDKs are used directly so Civica is not coupled to provider-specific
LangChain integrations. LangGraph remains the orchestration layer.
"""
from __future__ import annotations

from typing import TypeVar

from config import settings

T = TypeVar("T")

_GOOGLE_LEGACY_MODEL_MAP = {
    "gemini-2.5-flash": "gemini-3.6-flash",
    "models/gemini-2.5-flash": "gemini-3.6-flash",
}


def _google_client():
    from google import genai

    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for the configured Google model.")
    return genai.Client(api_key=settings.google_api_key)


def _openai_client():
    from openai import OpenAI

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the configured OpenAI model.")
    return OpenAI(api_key=settings.openai_api_key)


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


def generate_structured(
    *,
    task: str,
    prompt: str,
    schema: type[T],
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
    temperature: float = 0.2,
) -> T:
    provider, model = _model_config(task)

    if provider == "google":
        from google.genai import types

        client = _google_client()
        contents: list[object] = [prompt]
        if image_bytes is not None:
            if not image_mime_type:
                raise ValueError("image_mime_type is required when image_bytes is supplied.")
            contents = [
                types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type),
                prompt,
            ]

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return schema.model_validate_json(response.text)

    if provider == "openai":
        client = _openai_client()
        response = client.responses.parse(
            model=model,
            input=prompt,
            text_format=schema,
        )
        if response.output_parsed is None:
            raise RuntimeError("OpenAI returned no structured output.")
        return response.output_parsed

    raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_text(
    *,
    task: str,
    prompt: str,
    temperature: float = 0.4,
) -> str:
    provider, model = _model_config(task)

    if provider == "google":
        client = _google_client()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text or ""

    if provider == "openai":
        client = _openai_client()
        response = client.responses.create(
            model=model,
            input=prompt,
        )
        return response.output_text or ""

    raise ValueError(f"Unsupported LLM provider: {provider}")


def model_info(task: str) -> dict[str, str]:
    provider, model = _model_config(task)
    return {"provider": provider, "model": model}
