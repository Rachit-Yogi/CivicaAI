"""LangChain AI pipelines shared by API clients."""
from __future__ import annotations
import base64
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from llm import get_model
from schemas import ChatResponse, FraudAnalysis, SchemeAnalysis

SCHEME_SYSTEM = """You are Civica AI, an assistant for Indian government services and digital literacy. Analyze only supplied evidence. Never invent scheme rules. When evidence is insufficient, say so explicitly."""
FRAUD_SYSTEM = """You are Civica AI's fraud and misinformation triage assistant. Do not call content verified without evidence. Prefer Unverified___Needs_Caution when verification is unavailable."""
CHAT_SYSTEM = """You are Mitra, Civica AI's digital literacy assistant. Give clear practical answers and distinguish facts from uncertainty. Never fabricate official claims."""

def analyze_scheme(*, text: str | None = None, image_path: str | None = None, source_urls: list[str] | None = None) -> dict:
    source_urls = source_urls or []
    model = get_model("multimodal" if image_path else "text", temperature=0.1)
    if image_path:
        ext = Path(image_path).suffix.lower()
        mime = {".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",".webp":"image/webp"}[ext]
        data = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        message = HumanMessage(content=[{"type":"text","text":SCHEME_SYSTEM+" Identify the scheme and explain eligibility, benefits and application process."},{"type":"image_url","image_url":{"url":f"data:{mime};base64,{data}"}}])
        result = model.with_structured_output(SchemeAnalysis).invoke([message])
    else:
        prompt = ChatPromptTemplate.from_messages([("system",SCHEME_SYSTEM),("human","Sources: {sources}\nEvidence:\n{evidence}")])
        result = (prompt | model.with_structured_output(SchemeAnalysis)).invoke({"sources":source_urls,"evidence":text or "No evidence supplied."})
    payload = result.model_dump()
    payload["sources"] = list(dict.fromkeys(payload.get("sources", []) + source_urls))
    return payload

def analyze_fraud(*, text: str | None = None, image_path: str | None = None) -> dict:
    model = get_model("multimodal" if image_path else "text", temperature=0.0)
    if image_path:
        ext = Path(image_path).suffix.lower(); mime = {".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",".webp":"image/webp"}[ext]
        data = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        message = HumanMessage(content=[{"type":"text","text":FRAUD_SYSTEM},{"type":"text","text":text or "No additional text."},{"type":"image_url","image_url":{"url":f"data:{mime};base64,{data}"}}])
        result = model.with_structured_output(FraudAnalysis).invoke([message])
    else:
        prompt = ChatPromptTemplate.from_messages([("system",FRAUD_SYSTEM),("human","Content:\n{content}")])
        result = (prompt | model.with_structured_output(FraudAnalysis)).invoke({"content":text or ""})
    return result.model_dump()

def chat_reply(message: str, history: list[dict] | None = None) -> str:
    model = get_model("chat", temperature=0.4)
    msgs = [("system", CHAT_SYSTEM)]
    for item in (history or [])[-12:]: msgs.append(("assistant" if item.get("role") in {"assistant","model"} else "human", str(item.get("text",""))))
    msgs.append(("human", message))
    result = (ChatPromptTemplate.from_messages(msgs) | model.with_structured_output(ChatResponse)).invoke({})
    return result.response
