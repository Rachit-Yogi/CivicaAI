"""Civica AI FastAPI service."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ingestion import fetch_url_text, is_image, read_pdf, read_text, save_upload
from llm import model_info
from pipelines import analyze_fraud as run_fraud
from pipelines import analyze_scheme as run_scheme
from pipelines import chat_reply

load_dotenv()

app = FastAPI(title="Civica AI API", description="FastAPI gateway for Civica's LangChain-powered civic AI services.", version="2.0.0")

origins = [x.strip() for x in os.getenv("CIVICA_CORS_ORIGINS", "http://localhost:5000").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["*"])

class ApiResponse(BaseModel):
    success: bool = True
    data: dict

class SchemeRequest(BaseModel):
    text: str | None = Field(default=None, max_length=200_000)
    url: str | None = None

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=12)

class HealthResponse(BaseModel):
    status: str
    models: dict[str, dict[str, str]]


def _cleanup(path: str | None) -> None:
    if path:
        try:
            os.remove(path)
        except OSError:
            pass

@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", models={task: model_info(task) for task in ("text", "multimodal", "chat", "reasoning")})

@app.post("/api/v1/schemes/analyze", response_model=ApiResponse, tags=["schemes"])
def analyze_scheme(request: SchemeRequest) -> ApiResponse:
    if not request.text and not request.url:
        raise HTTPException(status_code=400, detail="Provide either text or url.")
    try:
        if request.url:
            text, sources = fetch_url_text(request.url)
            data = run_scheme(text=text, source_urls=sources)
        else:
            data = run_scheme(text=request.text)
        return ApiResponse(data=data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

@app.post("/api/v1/schemes/analyze-file", response_model=ApiResponse, tags=["schemes"])
async def analyze_scheme_file(file: UploadFile = File(...)) -> ApiResponse:
    temp_path = None
    try:
        temp_path = save_upload(file)
        suffix = Path(temp_path).suffix.lower()
        if is_image(temp_path):
            data = run_scheme(image_path=temp_path)
        elif suffix == ".pdf":
            data = run_scheme(text=read_pdf(temp_path))
        else:
            data = run_scheme(text=read_text(temp_path))
        return ApiResponse(data=data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        _cleanup(temp_path)

@app.post("/api/v1/fraud/analyze", response_model=ApiResponse, tags=["fraud"])
async def analyze_fraud(text: Annotated[str | None, Form(max_length=20_000)] = None, image: UploadFile | None = File(default=None)) -> ApiResponse:
    if not text and image is None:
        raise HTTPException(status_code=400, detail="Provide text or an image.")
    temp_path = None
    try:
        if image is not None:
            temp_path = save_upload(image)
            if not is_image(temp_path):
                raise HTTPException(status_code=400, detail="Fraud image must be PNG, JPEG, or WebP.")
        data = run_fraud(text=text, image_path=temp_path)
        return ApiResponse(data=data)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        _cleanup(temp_path)

@app.post("/api/v1/chat", response_model=dict, tags=["chat"])
def chat(request: ChatRequest) -> dict:
    try:
        return {"success": True, "response": chat_reply(request.message, request.history)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
