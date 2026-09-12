"""Civica AI workflows orchestrated with LangGraph and grounded model calls."""
from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from llm import generate_structured, generate_text
from schemas import FraudAnalysis, SchemeAnalysis

SCHEME_SYSTEM = """
You are Civica AI's Indian government scheme research assistant.

PRIMARY TASK
Identify the scheme the user is asking about and return a useful, concrete answer.
The user's input may be only a scheme name (for example, "PM CARES Fund"), a short question,
a pasted government document, an image, or an official URL.

GROUNDING RULES
1. Treat live retrieved evidence and explicitly supplied source material as the primary authority.
2. For scheme-name or question-only inputs, use Google Search grounding before answering.
3. Prioritize official sources: Government of India ministries/departments, official .gov.in sites,
   myScheme.gov.in, PIB, official state-government portals, and official scheme PDFs/notifications.
4. Prefer the most recent authoritative source when rules, amounts, dates, eligibility, or application
   procedures may have changed.
5. Do not invent a benefit, amount, eligibility condition, deadline, portal, or document.
6. Do not say "insufficient evidence provided" merely because the user's input is short. A short scheme
   name is a request to research and explain that scheme.
7. When multiple similarly named schemes exist, identify the best match and explain the distinction.

RESPONSE QUALITY
- Give concrete facts, not vague summaries.
- Explain what the scheme is for, who it is for, what the beneficiary receives, and how to access it.
- Include amounts, limits, dates, eligibility thresholds, exceptions, documents, helplines, and official
  application links when supported.
- Use numbered steps for application guidance.
- Use plain language suitable for a first-time citizen.
- Distinguish a fund/program/policy from a citizen-benefit scheme when that distinction matters.
- Never turn unknowns into guesses.
- If a particular field cannot be verified, say exactly what the official source does and does not state;
  do not use generic filler such as "information not available in the provided evidence."
- Return actual URLs in the sources field.
""".strip()

FRAUD_SYSTEM = """
You are Civica AI's fraud and misinformation triage assistant.
Use supplied evidence and, when available, current grounded sources. Do not call content verified without
adequate evidence. Prefer Unverified___Needs_Caution when verification cannot be established.
Return a practical explanation and the evidence-based reason for the classification.
""".strip()

CHAT_SYSTEM = """
You are Mitra, Civica AI's digital literacy assistant.
Give clear, practical answers. For current government/service facts, prefer grounded official information
and clearly distinguish verified facts from uncertainty. Never fabricate official claims.
""".strip()


class WorkflowState(TypedDict, total=False):
    operation: str
    prompt: str
    schema: type[Any]
    image_bytes: bytes | None
    image_mime_type: str | None
    temperature: float
    grounded: bool
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
            grounded=state.get("grounded", False),
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
    grounded: bool = False,
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
            "grounded": grounded,
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
    query = (text or "").strip()
    supplied_sources = "\n".join(source_urls) if source_urls else "No explicit URLs supplied."

    if image_path:
        prompt = (
            f"{SCHEME_SYSTEM}\n\nSUPPLIED URLS:\n{supplied_sources}\n\n"
            "IMAGE TASK:\nIdentify the scheme/document in the supplied image. Use grounded search to verify the "
            "scheme and retrieve its current official details. Extract concrete facts into every applicable "
            "response section."
        )
    else:
        prompt = (
            f"{SCHEME_SYSTEM}\n\nUSER REQUEST:\n{query}\n\n"
            f"SUPPLIED URLS:\n{supplied_sources}\n\n"
            "RESEARCH INSTRUCTION:\nIf the request is only a scheme name, research that scheme now using "
            "Google Search grounding. Search multiple authoritative sources when useful and synthesize the "
            "current, citizen-facing answer. Do not require the user to paste evidence before researching."
        )

    result = _run_structured(
        prompt=prompt,
        schema=SchemeAnalysis,
        temperature=0.1,
        image_path=image_path,
        grounded=True,
    )
    payload = result.model_dump()
    payload["sources"] = list(dict.fromkeys(payload.get("sources", []) + source_urls))
    return payload


def analyze_fraud(*, text: str | None = None, image_path: str | None = None) -> dict:
    prompt = f"{FRAUD_SYSTEM}\n\nContent to analyze:\n{text or 'No additional text supplied.'}"
    result = _run_structured(
        prompt=prompt,
        schema=FraudAnalysis,
        temperature=0.0,
        image_path=image_path,
        grounded=True,
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
