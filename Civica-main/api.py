"""Civica AI FastAPI application: API + web UI in one service."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from ingestion import fetch_url_text, is_image, read_pdf, read_text, save_upload
from llm import model_info
from pipelines import analyze_fraud as run_fraud
from pipelines import analyze_scheme as run_scheme
from pipelines import chat_reply

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Civica AI",
    description="FastAPI application and API gateway for Civica's civic AI services.",
    version="4.0.0",
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

origins = [x.strip() for x in os.getenv("CIVICA_CORS_ORIGINS", "http://localhost:8000").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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


FAVICON_SVG = """<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\">
<rect width=\"64\" height=\"64\" rx=\"14\" fill=\"#0f766e\"/>
<path d=\"M18 20h28v7H25v7h18v7H25v10h-7V20z\" fill=\"white\"/>
</svg>"""


def _cleanup(path: str | None) -> None:
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Web UI routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/index/", response_class=HTMLResponse, include_in_schema=False)
def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/home/", response_class=HTMLResponse, include_in_schema=False)
def home_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/scheme/", response_class=HTMLResponse, include_in_schema=False)
def scheme_page(request: Request):
    return templates.TemplateResponse("scheme.html", {"request": request})


@app.get("/fraud/", response_class=HTMLResponse, include_in_schema=False)
def fraud_page(request: Request):
    return templates.TemplateResponse("fraud.html", {"request": request})


@app.get("/mitra/", response_class=HTMLResponse, include_in_schema=False)
def mitra_page(request: Request):
    return templates.TemplateResponse("mitra.html", {"request": request})


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(content=FAVICON_SVG, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models={task: model_info(task) for task in ("text", "multimodal", "chat", "reasoning")},
    )


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
async def analyze_fraud(
    text: Annotated[str | None, Form(max_length=20_000)] = None,
    image: UploadFile | None = File(default=None),
) -> ApiResponse:
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
