# 🇮🇳 CIVICA AI – Voice-First Civic Intelligence Platform

CIVICA AI is a **voice-first, multilingual AI platform** designed to make government services, digital tools, and everyday information **accessible to non-tech users in Bharat**.

It enables users to interact using **voice, text, or images** and receive **simple, structured, and actionable responses**.

## 🚀 Problem Statement

Millions of people in India struggle with complex government portals, language barriers, low digital literacy, and rising scams and misinformation. CIVICA AI addresses these problems with a simple, multimodal civic assistant.

## 💡 Key Features

- 🎤 Voice-first interaction (product direction)
- 🏛 Government scheme guidance: eligibility, benefits, and application steps
- 🛡 Fake news and scam detection for text and images
- 📚 Digital literacy support through Mitra AI
- 🌐 Multimodal input for civic workflows

## 🏗 Architecture

```text
Browser
   │
   ▼
FastAPI (web UI + REST API)
   │
   ├── Scheme analysis
   ├── Fraud / scam analysis
   └── Mitra chat
   │
   ▼
LangChain pipelines
   │
   ├── Gemini 2.5 Flash (text / multimodal / chat)
   └── OpenAI reasoning model (configurable)
```

FastAPI is the **single application server**. The legacy Flask application has been removed.

## 🛠 Tech Stack

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Python 3
- FastAPI
- Uvicorn
- Pydantic

### AI
- LangChain
- Gemini 2.5 Flash
- OpenAI (configurable reasoning route)

### Document / Web Processing
- pypdf
- Pillow
- BeautifulSoup4
- requests

## ▶️ Run Locally

From `Civica-main/`:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r ../requirements.txt
```

Create `.env` from `.env.example` and set `GOOGLE_API_KEY`.

Start the application:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Open:
- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/scheme/`
- `http://127.0.0.1:8000/fraud/`
- `http://127.0.0.1:8000/mitra/`
- `http://127.0.0.1:8000/docs`

## 🔌 API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Health + model configuration |
| POST | `/api/v1/schemes/analyze` | Analyze text or a public URL |
| POST | `/api/v1/schemes/analyze-file` | Analyze PDF, text, or image upload |
| POST | `/api/v1/fraud/analyze` | Analyze suspicious text or image |
| POST | `/api/v1/chat` | Mitra conversational assistant |

## 🧪 Testing

Run from the repository root:

```bash
pytest Civica-main/tests -q
```

The smoke suite covers API routing, validation, UI routes, template presence, and frontend-to-FastAPI endpoint contracts.

## 📌 Current Status

- ✅ FastAPI is the primary application server
- ✅ Flask removed from runtime dependencies
- ✅ LangChain model-routing layer integrated
- ✅ Web UI migrated to `/api/v1/*` endpoints
- 🔄 Voice, OCR, retrieval, and mobile clients remain extension areas

## 🎯 Vision

Make technology accessible, understandable, and useful for every citizen of India — regardless of language or technical knowledge.

## 🤝 Contributing

Contributions, ideas, and feedback are welcome.

## 📬 Contact

**Rachit Yogi**  
AI & Data Science Enthusiast
