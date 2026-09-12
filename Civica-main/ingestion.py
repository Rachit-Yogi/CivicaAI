"""Safe ingestion for text, PDF, image and public URLs."""
from __future__ import annotations
import ipaddress, mimetypes, os, shutil, socket, tempfile
from pathlib import Path
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from config import settings

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_MIME = {"application/pdf", "text/plain", "image/png", "image/jpeg", "image/webp"}

def _assert_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public http(s) URLs are supported.")
    try: ip = ipaddress.ip_address(socket.gethostbyname(parsed.hostname))
    except (socket.gaierror, ValueError) as exc: raise ValueError("The URL host could not be resolved.") from exc
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved: raise ValueError("Private or local network URLs are not allowed.")

def fetch_url_text(url: str) -> tuple[str, list[str]]:
    _assert_public_url(url)
    response = requests.get(url, timeout=settings.request_timeout_seconds, headers={"User-Agent": "CivicaAI/2.0"})
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").split(";")[0].lower()
    if content_type not in {"text/html", "text/plain"}: raise ValueError("URL must return HTML or plain text content.")
    if content_type == "text/html":
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]): tag.decompose()
        text = "\n".join(x.strip() for x in soup.get_text("\n").splitlines() if x.strip())
    else: text = response.text
    return text[:200_000], [url]

def read_pdf(path: str) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)[:200_000]

def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")[:200_000]

def save_upload(file_storage) -> str:
    filename = Path(getattr(file_storage, "filename", None) or "upload.txt").name
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS: raise ValueError(f"Unsupported file type: {ext or 'unknown'}")
    stream = getattr(file_storage, "stream", None) or getattr(file_storage, "file", None) or file_storage
    stream.seek(0, os.SEEK_END); size = stream.tell(); stream.seek(0)
    if size > settings.max_upload_mb * 1024 * 1024: raise ValueError(f"File exceeds the {settings.max_upload_mb} MB limit.")
    fd, path = tempfile.mkstemp(prefix="civica_", suffix=ext); os.close(fd)
    try:
        if hasattr(file_storage, "save"): file_storage.save(path)
        else:
            with open(path, "wb") as out: shutil.copyfileobj(stream, out)
        mime = mimetypes.guess_type(path)[0]
        if mime and mime not in ALLOWED_MIME: raise ValueError("File MIME type is not allowed.")
        return path
    except Exception:
        try: os.remove(path)
        except OSError: pass
        raise

def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
