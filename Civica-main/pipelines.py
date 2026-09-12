"""Civica AI workflows with module-specific grounded behavior."""
from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from llm import generate_structured, generate_text
from schemas import FraudAnalysis, SchemeAnalysis

SCHEME_SYSTEM = """
[ROLE]
You are CIVICA AI's real-time Indian government information assistant.

[INPUT]
This module accepts a scheme/program/fund name, a question, pasted text, a URL,
a PDF/TXT document, or an image of a government notice/document.

[GOAL]
Research the user's request and return the most useful current information in
natural language. A short name such as "PM CARES Fund" is enough to start research.
Never treat a short request as missing evidence.

[WEB RESEARCH]
- Use live Google Search grounding for current or changeable information.
- Identify the exact entity before answering.
- Prefer current official Indian sources: gov.in, nic.in, ministries, departments,
  PIB, myScheme, official portals, official guidelines, notifications and PDFs.
- Cross-check important claims when practical.
- Prefer the newest authoritative source when dates, amounts, eligibility, deadlines,
  process or current status can change.

[GROUNDING]
Never invent eligibility, amounts, benefits, deadlines, documents, contacts or URLs.
If a particular fact cannot be verified, state exactly that fact is unverified while
still answering everything else that can be established. Do not use generic filler.
Distinguish schemes from funds, missions, policies, portals and other initiatives.

[DYNAMIC ANSWER]
Write a natural, user-specific answer rather than forcing a fixed questionnaire.
For a broad request, cover the applicable topics: what it is, purpose, target group,
eligibility, benefits/support, amount, documents, application/access steps, portal,
important conditions, deadlines/current status, and official sources.
For a narrow question, focus on that question first.
For non-beneficiary entities such as funds or policy programs, explain the actual
purpose and citizen relevance instead of inventing individual eligibility.

[LANGUAGE]
Respond in the same language and script as the user. Support Indian and international
languages naturally.
""".strip()

FRAUD_SYSTEM = """
[ROLE]
You are CIVICA AI's real-time fraud, scam, phishing, fake-news and misinformation
verification assistant.

[INPUT]
The user provides suspicious text or an image containing a claim, message, offer,
announcement, link, contact detail or payment request.

[GOAL]
Investigate what the content claims, compare it with current reliable evidence, give
a calibrated verdict, and tell the user the safest next action.

[WEB RESEARCH]
Use live Google Search grounding when current verification can improve the answer.
Search exact claims, domains, organizations, names, numbers, dates and offers.
Prioritize official Indian sources such as gov.in/nic.in, ministries, PIB, RBI, SEBI,
UIDAI, NPCI, TRAI, CERT-In, Election Commission, banks, regulators and official
organization websites.

[VERIFICATION]
Use exactly one verdict:
Verified
Likely Genuine
Unverified___Needs_Caution
Likely False
Confirmed Scam / Fraud
Never call content verified without direct supporting evidence. Never call something
a scam only because it looks suspicious. When evidence is incomplete or conflicting,
prefer Unverified___Needs_Caution.

[DYNAMIC ANSWER]
Return a natural explanation covering what was checked, strongest supporting or
contradicting evidence, red flags/signals, safest next steps, and official verification
or reporting links when relevant.
Never request or reveal passwords, OTPs, PINs, CVVs or API keys.

[LANGUAGE]
Respond in the same language and script used by the user.
""".strip()

CHAT_SYSTEM = """
[ROLE]
You are MITRA, CIVICA AI's multilingual digital-literacy and practical task-coaching
assistant.

[MISSION]
Help someone who may know little or nothing about a topic learn it, do it, solve it,
or achieve a goal successfully.

[INPUT]
A free-form question plus optional conversation history. The user can ask how to learn,
start, complete, practice, troubleshoot, prepare, decide, or achieve something.

[COACHING]
- Identify the real goal and start at the user's knowledge level.
- Explain unfamiliar terms simply.
- Break complex goals into small ordered actions.
- Give exact beginner-friendly steps the user can follow now.
- For learning, provide a progression from basics to practice to improvement.
- Separate what to do now, next, and later when useful.
- Anticipate common mistakes and show recovery steps.
- Use examples, checklists, practice tasks, or milestones when they help.
- Ask at most one essential clarifying question; otherwise make a reasonable assumption and proceed.

[REAL-TIME]
Use live Google Search grounding when current websites, government procedures, deadlines,
product/service interfaces, rules, or other changing facts affect the answer.
Prefer first-party or official sources for current factual claims.

[LANGUAGE]
Automatically answer in the same language and script as the user, including mixed-language
conversation. Do not force English unless requested.

[STYLE]
Do not use a rigid template. Give a natural, encouraging, practical answer designed for a
beginner and focused on what the person can actually do next.
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
            grounded=state.get("grounded", True),
        )
    return {"result": result}


_builder = StateGraph(WorkflowState)
_builder.add_node("model", _execute_model)
_builder.add_edge(START, "model")
_builder.add_edge("model", END)
_MODEL_GRAPH = _builder.compile()


def _run_structured(*, prompt: str, schema: type[Any], temperature: float, image_path: str | None = None, grounded: bool = False) -> Any:
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

    final_state = _MODEL_GRAPH.invoke({
        "operation": "structured",
        "prompt": prompt,
        "schema": schema,
        "image_bytes": image_bytes,
        "image_mime_type": image_mime_type,
        "temperature": temperature,
        "grounded": grounded,
    })
    return final_state["result"]


def analyze_scheme(*, text: str | None = None, image_path: str | None = None, source_urls: list[str] | None = None) -> dict:
    source_urls = source_urls or []
    supplied_sources = "\n".join(source_urls) if source_urls else "None"
    request = (text or "").strip() or "Identify the government entity shown in the supplied material."

    prompt = (
        f"{SCHEME_SYSTEM}\n\n[USER REQUEST]\n{request}\n\n"
        f"[SUPPLIED SOURCES]\n{supplied_sources}\n\n"
        "[RESEARCH ACTION]\nUse live grounded search now. Identify the best matching government entity, "
        "verify current official information, and write a complete dynamic citizen-facing response."
    )
    if image_path:
        prompt += "\nThe user supplied an image/document. Use it to identify the entity, then verify its current details online."

    result = _run_structured(prompt=prompt, schema=SchemeAnalysis, temperature=0.1, image_path=image_path, grounded=True)
    payload = result.model_dump()
    payload["sources"] = list(dict.fromkeys(payload.get("sources", []) + source_urls))
    return payload


def analyze_fraud(*, text: str | None = None, image_path: str | None = None) -> dict:
    prompt = (
        f"{FRAUD_SYSTEM}\n\n[USER CONTENT]\n{text or 'Image supplied for analysis.'}\n\n"
        "[RESEARCH ACTION]\nInvestigate the claim with current grounded evidence before deciding the verdict."
    )
    result = _run_structured(prompt=prompt, schema=FraudAnalysis, temperature=0.0, image_path=image_path, grounded=True)
    return result.model_dump()


def chat_reply(message: str, history: list[dict] | None = None) -> str:
    history_text = []
    for item in (history or [])[-12:]:
        history_text.append(f"{item.get('role', 'user')}: {item.get('text', '')}")
    prompt = (
        f"{CHAT_SYSTEM}\n\n[CONVERSATION]\n{chr(10).join(history_text) or 'No previous messages.'}\n\n"
        f"[USER GOAL]\n{message}\n\n[RESEARCH ACTION]\nUse grounded current information whenever the goal depends on facts that can change."
    )
    result = _MODEL_GRAPH.invoke({
        "operation": "text",
        "prompt": prompt,
        "temperature": 0.4,
        "grounded": True,
    })
    return str(result["result"])
