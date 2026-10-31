"""FastAPI app: KB upload, RAG chat with escalation, and support tickets. API docs at /docs.

The chat flow itself is the LangGraph agent in agent.py; this module is the HTTP layer.
"""
import json
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

import agent
import db
import kb
import llm  # fails fast with a clear message if GEMINI_API_KEY is missing

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("support")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

db.init()
app = FastAPI(title="AI Customer Support Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StrictModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)


class ConversationIn(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=200)


class ChatIn(StrictModel):
    conversation_id: int
    message: str = Field(min_length=1, max_length=2000)


class TicketIn(StrictModel):
    conversation_id: int
    summary: str | None = Field(default=None, min_length=1, max_length=500)
    category: str | None = None


class TicketStatusIn(BaseModel):
    status: Literal["Open", "In Progress", "Resolved"]


# ---------- Knowledge base ----------

@app.post("/kb/upload", status_code=201)
def upload_document(file: UploadFile):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in kb.SUPPORTED_TYPES:
        raise HTTPException(400, f"Unsupported file type '{ext or 'none'}'. Upload a PDF, TXT or DOCX file.")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is too large (max 10 MB).")

    doc_id = db.execute(
        "INSERT INTO documents (filename, uploaded_at, status) VALUES (?, ?, 'Processing')",
        (file.filename, db.now()),
    )
    try:
        chunks = kb.chunk_text(kb.extract_text(file.filename, data))
        kb.add_document(doc_id, file.filename, chunks)
    except kb.ExtractionError as e:
        db.execute("UPDATE documents SET status = 'Failed', error = ? WHERE id = ?", (str(e), doc_id))
        raise HTTPException(422, str(e))
    except Exception as e:
        log.exception("Ingestion failed for %s", file.filename)
        db.execute("UPDATE documents SET status = 'Failed', error = ? WHERE id = ?", (str(e), doc_id))
        raise HTTPException(500, f"Processing failed: {e}")
    db.execute("UPDATE documents SET status = 'Processed', chunk_count = ? WHERE id = ?", (len(chunks), doc_id))
    return db.query_one("SELECT * FROM documents WHERE id = ?", (doc_id,))


@app.get("/kb/documents")
def list_documents():
    return db.query("SELECT * FROM documents ORDER BY id DESC")


# ---------- Conversations & chat ----------

def get_conversation(conversation_id: int) -> dict:
    conv = db.query_one("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
    if not conv:
        raise HTTPException(404, "Conversation not found. Start a new conversation.")
    return conv


@app.post("/conversations", status_code=201)
def create_conversation(body: ConversationIn):
    conv_id = db.execute(
        "INSERT INTO conversations (customer_name, customer_email, created_at) VALUES (?, ?, ?)",
        (body.name, body.email, db.now()),
    )
    return get_conversation(conv_id)


@app.get("/conversations/{conversation_id}")
def read_conversation(conversation_id: int):
    return {**get_conversation(conversation_id), "messages": db.get_messages(conversation_id)}


@app.post("/chat")
def chat(body: ChatIn):
    """Runs the agent and returns only the final result (non-streaming)."""
    conv = get_conversation(body.conversation_id)
    for event in agent.chat_events(conv, body.message):
        pass
    return event


@app.post("/chat/stream")
def chat_stream(body: ChatIn):
    """Same as /chat, streamed as JSON Lines: "token" events, then one "done" (or "error") event."""
    conv = get_conversation(body.conversation_id)  # a 404 here is still a normal HTTP error

    def lines():
        try:
            for event in agent.chat_events(conv, body.message):
                yield json.dumps(event) + "\n"
        except HTTPException as e:  # status is already 200 once streaming starts
            yield json.dumps({"type": "error", "detail": e.detail}) + "\n"

    return StreamingResponse(lines(), media_type="application/jsonl")


# ---------- Tickets ----------

@app.post("/tickets", status_code=201)
def post_ticket(body: TicketIn):
    if body.category and body.category not in llm.CATEGORIES:
        raise HTTPException(422, f"Category must be one of: {', '.join(llm.CATEGORIES)}")
    conv = get_conversation(body.conversation_id)
    result = agent.escalate(conv, agent.TICKET_REQUESTED, body.category, body.summary)
    return get_ticket(result["ticket_id"])


@app.get("/tickets")
def list_tickets():
    return db.query("SELECT * FROM tickets ORDER BY id DESC")


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: int):
    ticket = db.query_one("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    if not ticket:
        raise HTTPException(404, "Ticket not found")
    return {**ticket, "messages": db.get_messages(ticket["conversation_id"])}


@app.patch("/tickets/{ticket_id}")
def update_ticket_status(ticket_id: int, body: TicketStatusIn):
    get_ticket(ticket_id)
    db.execute("UPDATE tickets SET status = ? WHERE id = ?", (body.status, ticket_id))
    return get_ticket(ticket_id)


# ---------- Frontend ----------

# The built React app (made by the Dockerfile, or by `npm run build`), served on the API's port.
# FastAPI checks the API routes first, so these files never shadow an endpoint.
FRONTEND_BUILD = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_BUILD.is_dir():
    app.frontend("/", directory=FRONTEND_BUILD)
