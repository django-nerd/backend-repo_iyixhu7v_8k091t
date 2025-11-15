import os
import io
import json
import logging
import tempfile
from datetime import datetime, timezone
from typing import Optional, List, Literal

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from database import db, create_document, get_documents

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("mindcraft")

app = FastAPI(title="MindCraft AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auth (Supabase JWT compatible simple verification) ---
security = HTTPBearer()
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "dev-secret")

import jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256", "RS256"], options={"verify_aud": False})
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- Schemas ---
class ProcessResponse(BaseModel):
    document_id: str
    title: str
    type: Literal['pdf', 'youtube']

class AccuracyPayload(BaseModel):
    user_id: str
    question_id: str
    correct: bool

class SemanticSearchPayload(BaseModel):
    document_id: str
    query: str

# Helper: DB collection names
COL_USERS = "users"
COL_DOCS = "documents"
COL_NOTES = "notes"
COL_QUIZZES = "quizzes"
COL_FLASHCARDS = "flashcards"
COL_ACCURACY = "accuracy_tracking"
COL_SEMANTIC = "semantic_index"

# --- Utilities ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

from openai import OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

PROMPT = (
    "You are an educational content transformer. Given the raw lecture/PDF text, generate:\n"
    "1. Highly structured notes with headings.\n"
    "2. 8–12 MCQs with four options each, correct answer, and short explanation.\n"
    "3. 5–10 True/False questions with reasoning.\n"
    "4. 20 Flashcards in simple Q/A format.\n"
    "Ensure information is factually correct, concise, and not hallucinated. Do NOT create information not found in the source text."
)

# --- Audio utils ---
from pydub import AudioSegment
import subprocess

def compress_audio_to_opus(input_path: str, output_path: str):
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path, "-c:a", "libopus", "-b:a", "16k", output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception as e:
        logger.warning(f"ffmpeg compression failed, fallback to pydub mp3: {e}")
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="mp3", bitrate="16k")

# --- Input processing ---
import yt_dlp
from pypdf import PdfReader


def transcribe_with_whisper(audio_path: str) -> str:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    with open(audio_path, "rb") as f:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript.text


def download_youtube_audio(url: str, out_dir: str) -> str:
    ydl_opts = {
        'format': 'bestaudio[abr<=64]',
        'outtmpl': os.path.join(out_dir, '%(id)s.%(ext)s'),
        'postprocessors': [],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    return filename


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def chunk_text(s: str, max_chars: int = 8000) -> List[str]:
    chunks = []
    cur = 0
    while cur < len(s):
        chunks.append(s[cur:cur+max_chars])
        cur += max_chars
    return chunks

# --- AI generation ---

def call_openai_chat(text: str) -> dict:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    completion = openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role":"system","content":PROMPT},{"role":"user","content":text}],
        temperature=0.2
    )
    content = completion.choices[0].message.content
    return safe_parse_ai_output(content)


def safe_parse_ai_output(content: str) -> dict:
    try:
        return json.loads(content)
    except Exception:
        return {
            "notes": content,
            "mcqs": [],
            "true_false": [],
            "flashcards": []
        }


def create_embeddings(texts: List[str]) -> List[List[float]]:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    resp = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    import math
    dot = sum(a*b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a*a for a in v1))
    n2 = math.sqrt(sum(b*b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)

# --- Business flow ---

def generate_all_from_text(raw_text: str) -> dict:
    result = call_openai_chat(raw_text[:16000])
    result.setdefault("notes", "")
    result.setdefault("mcqs", [])
    result.setdefault("true_false", [])
    result.setdefault("flashcards", [])
    return result


# --- Endpoints ---
@app.post("/process-pdf", response_model=ProcessResponse)
async def process_pdf(background: BackgroundTasks, file: UploadFile = File(...), title: Optional[str] = Form(None), user=Depends(verify_token)):
    try:
        data = await file.read()
        raw_text = extract_pdf_text(data)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")
        doc = {
            "user_id": user.get("sub") or user.get("user_id"),
            "title": title or file.filename,
            "type": "pdf",
            "raw_text": raw_text,
            "created_at": datetime.now(timezone.utc)
        }
        document_id = create_document(COL_DOCS, doc)

        def bg_job(doc_id: str, text: str):
            try:
                out = generate_all_from_text(text)
                create_document(COL_NOTES, {"document_id": doc_id, "notes_json": out.get("notes")})
                create_document(COL_QUIZZES, {"document_id": doc_id, "mcqs_json": out.get("mcqs"), "tf_json": out.get("true_false")})
                create_document(COL_FLASHCARDS, {"document_id": doc_id, "flashcards_json": out.get("flashcards")})
                # embeddings
                notes_text = out.get("notes") if isinstance(out.get("notes"), str) else json.dumps(out.get("notes"))
                vec = create_embeddings([notes_text])[0]
                create_document(COL_SEMANTIC, {"document_id": doc_id, "vector_embedding": vec})
            except Exception as e:
                logger.error(f"BG job failed: {e}")

        background.add_task(bg_job, document_id, raw_text)
        return {"document_id": document_id, "title": doc["title"], "type": "pdf"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("process-pdf failed")
        raise HTTPException(status_code=500, detail=str(e))


class YouTubeIn(BaseModel):
    url: str
    title: Optional[str] = None

@app.post("/process-youtube", response_model=ProcessResponse)
async def process_youtube(background: BackgroundTasks, payload: YouTubeIn, user=Depends(verify_token)):
    try:
        url = payload.url
        if not (url.startswith("http://") or url.startswith("https://")) or 'youtube' not in url:
            raise HTTPException(status_code=400, detail="Invalid YouTube link")
        with tempfile.TemporaryDirectory() as td:
            downloaded = download_youtube_audio(url, td)
            # compress low bitrate
            compressed_path = os.path.join(td, "audio.opus")
            compress_audio_to_opus(downloaded, compressed_path)
            # transcribe
            text = transcribe_with_whisper(compressed_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No transcription produced")
        doc = {
            "user_id": user.get("sub") or user.get("user_id"),
            "title": payload.title or "YouTube Lecture",
            "type": "youtube",
            "raw_text": text,
            "created_at": datetime.now(timezone.utc)
        }
        document_id = create_document(COL_DOCS, doc)

        def bg_job(doc_id: str, raw: str):
            try:
                out = generate_all_from_text(raw)
                create_document(COL_NOTES, {"document_id": doc_id, "notes_json": out.get("notes")})
                create_document(CCOL_QUIZZES if False else COL_QUIZZES, {"document_id": doc_id, "mcqs_json": out.get("mcqs"), "tf_json": out.get("true_false")})
                create_document(COL_FLASHCARDS, {"document_id": doc_id, "flashcards_json": out.get("flashcards")})
                notes_text = out.get("notes") if isinstance(out.get("notes"), str) else json.dumps(out.get("notes"))
                vec = create_embeddings([notes_text])[0]
                create_document(COL_SEMANTIC, {"document_id": doc_id, "vector_embedding": vec})
            except Exception as e:
                logger.error(f"BG job failed: {e}")

        background.add_task(bg_job, document_id, text)
        return {"document_id": document_id, "title": doc["title"], "type": "youtube"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("process-youtube failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/notes/{document_id}")
async def get_notes(document_id: str, user=Depends(verify_token)):
    items = get_documents(COL_NOTES, {"document_id": document_id}, limit=1)
    return items[0] if items else {"notes_json": None}


@app.get("/quizzes/{document_id}")
async def get_quizzes(document_id: str, user=Depends(verify_token)):
    items = get_documents(COL_QUIZZES, {"document_id": document_id}, limit=1)
    return items[0] if items else {"mcqs_json": [], "tf_json": []}


@app.get("/flashcards/{document_id}")
async def get_flashcards(document_id: str, user=Depends(verify_token)):
    items = get_documents(COL_FLASHCARDS, {"document_id": document_id}, limit=1)
    return items[0] if items else {"flashcards_json": []}


@app.post("/accuracy")
async def post_accuracy(payload: AccuracyPayload, user=Depends(verify_token)):
    create_document(COL_ACCURACY, {
        "user_id": payload.user_id,
        "question_id": payload.question_id,
        "correct": payload.correct,
        "timestamp": datetime.now(timezone.utc)
    })
    return {"status": "ok"}


@app.get("/user/documents")
async def user_documents(user=Depends(verify_token)):
    uid = user.get("sub") or user.get("user_id")
    docs = get_documents(COL_DOCS, {"user_id": uid})
    for d in docs:
        d["id"] = str(d.get("_id"))
        d.pop("_id", None)
    return docs


@app.post("/semantic-search")
async def semantic_search(payload: SemanticSearchPayload, user=Depends(verify_token)):
    query_vec = create_embeddings([payload.query])[0]
    rec = get_documents(COL_SEMANTIC, {"document_id": payload.document_id}, limit=1)
    if not rec:
        return {"results": []}
    stored = rec[0].get("vector_embedding")
    sim = cosine_similarity(query_vec, stored)
    return {"results": [{"document_id": payload.document_id, "similarity": sim}]}


@app.get("/")
def root():
    return {"message": "MindCraft AI backend running"}


@app.get("/test")
def test_database():
    resp = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            resp["database"] = "✅ Connected"
            resp["connection_status"] = "Connected"
            try:
                resp["collections"] = db.list_collection_names()[:10]
                resp["database"] = "✅ Connected & Working"
            except Exception as e:
                resp["database"] = f"⚠️ Connected but Error: {e}"[:60]
    except Exception as e:
        resp["database"] = f"❌ Error: {e}"[:60]

    resp["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    resp["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
