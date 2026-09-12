"""Civica AI workflows orchestrated with LangGraph.

LangGraph handles the deterministic workflow boundary while the official
Google GenAI and OpenAI SDKs perform model calls directly.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from llm import generate_structured, generate_text
from schemas import ChatResponse, FraudAnalysis, SchemeAnalysis

SCHEME_SYSTEM = """You are Civica AI, an assistant for Indian government services and digital literacy.
Analyze only supplied evidence. Never invent scheme rules. When evidence is insufficient, say so explicitly.
Return a concise, useful explanation of the scheme, eligibility, benefits, application process, and sources.
"""

FRAUD_SYSTEM = """You are Civica AI's fraud and misinformation triage assistant.
Do not call content verified without evidence. Prefer Unverified___Needs_Caution when verification is unavailable.
Return a practical explanation and the evidence-based reason for the classification.
"""

CHAT_SYSTEM = """You are Mitra, Civica AI's digital literacy assistant.
Give clear, practical answers and distinguish facts from uncertainty. Never fabricate official claims.
"""


class WorkflowState(TypedDict, total=False):
    operation: str
    prompt: str
    schema: type[Any]
    image_bytes: bytes | None
    image_mime_type: str | None
    temperature: float
    result: Any


def _execute_model(state: WorkflowState) -> dict[str, Any]:
    if state["operation"] == "structured":
        result = generate_structured(
            task="multimodal" if state.get("image_bytes") else "text",
            prompt=state["prompt"],
            schema=state["schema"],
            image_bytes=state.get("image_bytes"),
            image_mime_type=state.get("image_mime_type"),
            temperature=state.get("temperature", 0.2),
        )
    else:
        result = generate_text(
            task="chat",
            prompt=state["prompt"],
            temperature=state.get("temperature", 0.4),
        )
    return {"result": result}


_builder = StateGraph(WorkflowState)
_builder.add_node("model", _execute_model)
_builder.add_edge(START, "model")
_builder.add_edge("model", END)
_MODEL_GRAPH = _builder.compile()


def _run_structured(
    *,
    prompt: str,
    schema: type[Any],
    temperature: float,
    image_path: str | None = None,
) -> Any:
    image_bytes = None
    image_mime_type = None
    if image_path:
        path = Path(image_path)
        image_bytes = path.read_bytes()
        image_mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(path.suffix.lower())
        if not image_mime_type:
            raise ValueError(f"Unsupported image type: {path.suffix}")

    final_state = _MODEL_GRAPH.invoke(
        {
            "operation": "structured",
            "prompt": prompt,
            "schema": schema,
            "image_bytes": image_bytes,
            "image_mime_type": image_mime_type,
            "temperature": temperature,
        }
    )
    return final_state["result"]


def analyze_scheme(
    *,
    text: str | None = None,
    image_path: str | None = None,
    source_urls: list[str] | None = None,
) -> dict:
    source_urls = source_urls or []
    if image_path:
        prompt = (
            SCHEME_SYSTEM
            + "\nIdentify the scheme visible in the supplied image and explain eligibility, benefits, "
            + "application process, and any official source links that are explicitly visible."
        )
    else:
        prompt = (
            f"{SCHEME_SYSTEM}\nSources: {source_urls}\nEvidence:\n"
            f"{text or 'No evidence supplied.'}"
        )

    result = _run_structured(
        prompt=prompt,
        schema=SchemeAnalysis,
        temperature=0.1,
        image_path=image_path,
    )
    payload = result.model_dump()
    payload["sources"] = list(dict.fromkeys(payload.get("sources", []) + source_urls))
    return payload


def analyze_fraud(*, text: str | None = None, image_path: str | None = None) -> dict:
    prompt = f"{FRAUD_SYSTEM}\nContent to analyze:\n{text or 'No additional text supplied.'}"
    result = _run_structured(
        prompt=prompt,
        schema=FraudAnalysis,
        temperature=0.0,
        image_path=image_path,
    )
    return result.model_dump()


def chat_reply(message: str, history: list[dict] | None = None) -> str:
    history_text = []
    for item in (history or [])[-12:]:
        role = item.get("role", "user")
        text = item.get("text", "")
        history_text.append(f"{role}: {text}")
    prompt = (
        f"{CHAT_SYSTEM}\nConversation history:\n"
        f"{chr(10).join(history_text) or 'No previous messages.'}\n\n"
        f"user: {message}"
    )
    return generate_text(task="chat", prompt=prompt, temperature=0.4)
