"""Provider/model factory using LangChain chat-model interfaces."""
from __future__ import annotations
from config import settings

def _google(model: str, temperature: float):
    from langchain_google_genai import ChatGoogleGenerativeAI
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for the configured Google model.")
    return ChatGoogleGenerativeAI(model=model, google_api_key=settings.google_api_key, temperature=temperature, max_retries=4)

def _openai(model: str, temperature: float):
    from langchain_openai import ChatOpenAI
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the configured OpenAI model.")
    return ChatOpenAI(model=model, api_key=settings.openai_api_key, temperature=temperature, max_retries=4)

def get_model(task: str, *, temperature: float = 0.2):
    if task == "multimodal": provider, model = settings.multimodal_provider, settings.multimodal_model
    elif task == "chat": provider, model = settings.chat_provider, settings.chat_model
    elif task == "reasoning": provider, model = "openai", settings.openai_reasoning_model
    else: provider, model = settings.text_provider, settings.text_model
    if provider == "google": return _google(model, temperature)
    if provider == "openai": return _openai(model, temperature)
    raise ValueError(f"Unsupported LLM provider: {provider}")

def model_info(task: str) -> dict[str, str]:
    if task == "multimodal": return {"provider": settings.multimodal_provider, "model": settings.multimodal_model}
    if task == "chat": return {"provider": settings.chat_provider, "model": settings.chat_model}
    if task == "reasoning": return {"provider": "openai", "model": settings.openai_reasoning_model}
    return {"provider": settings.text_provider, "model": settings.text_model}
