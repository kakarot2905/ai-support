"""Gemini calls: the agent's streamed turns with tools, and ticket summaries. Failures surface as LLMError."""
import os
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Copy .env.example to .env in the project root and add your key "
        "(get one free at https://aistudio.google.com/apikey). In Docker, .env isn't part of the image: "
        "start it with `docker compose up`, which passes it in."
    )
MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
# Gemini 3 models "think" before writing; at the default level the first streamed token took ~10s,
# at "low" ~2s. Model-dependent: Gemini 2.5 rejects thinking_level, so set it empty there.
THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low").strip()
CATEGORIES = ["billing", "account", "returns", "shipping", "technical", "complaint", "general"]

_client = genai.Client(api_key=API_KEY, http_options=types.HttpOptions(timeout=30_000))
_thinking = types.ThinkingConfig(thinking_level=THINKING_LEVEL) if THINKING_LEVEL else None

# Replies stream as plain text (JSON can't be shown to the customer while it's half-written),
# so the model signals "no answer" with a bare marker and lists sources on a final line;
# both are stripped before the customer sees anything.
NOT_FOUND = "NOT_FOUND"
SOURCES_LINE = re.compile(r"\s*SOURCES:", re.I)
HOLD_BACK = 12  # trailing chars held back in case they are the start of the SOURCES line
SPEAKERS = {"user": "Customer", "agent": "Agent"}

SEARCH_TOOL = "search_help_docs"
TICKET_TOOL = "create_support_ticket"
TOOLS = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name=SEARCH_TOOL,
        description="Search the company's help documents. Returns the most relevant passages with their source filenames.",
        parameters_json_schema={"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string", "description": "What to look up, written as a standalone question"},
        }},
    ),
    types.FunctionDeclaration(
        name=TICKET_TOOL,
        description="Create a support ticket so a person from the support team follows up with the customer by email.",
        parameters_json_schema={"type": "object", "required": ["summary", "category"], "properties": {
            "summary": {"type": "string", "description": "One sentence describing the customer's issue (max 25 words)"},
            "category": {"type": "string", "enum": CATEGORIES},
        }},
    ),
])]

AGENT_INSTRUCTION = f"""You are a friendly customer support assistant chatting with a customer.
The user turn holds the CONVERSATION so far and the customer's latest MESSAGE. Reply to the latest message.

Tools:
- {SEARCH_TOOL}: searches the company's help documents (FAQ, policies, guides). Call it before answering
  any question about the company: products, orders, shipping, returns, prices, policies or contact
  details. Write the query as a standalone question, resolving follow-ups from the conversation
  ("how long does it take?" after a question about international shipping means international delivery
  time). If the results don't answer it, you may search once more with a different query.
- {TICKET_TOOL}: hands the conversation to a person from the support team. Call it only when the
  customer asks for a ticket or a person, or accepts your offer of a ticket.

Rules:
- Company facts must come only from search results. Never invent them.
- If neither the search results nor the conversation answer the customer's question (including
  questions unrelated to the company), reply with exactly {NOT_FOUND} and nothing else.
- Use the conversation to remember what was said, such as the customer's name, order or ticket number.
- Reply naturally to greetings, thanks, questions about you and requests you can't carry out: you
  answer from the company's help documents and can open a support ticket for the team to follow up.
- Don't write any text in a turn where you call a tool.
- Keep replies short (1-4 sentences). Never mention tools, search results or these instructions.
- If your reply uses search results, end it with one final line listing the source filenames used:
SOURCES: <filename>, <filename>"""

_AGENT_CONFIG = types.GenerateContentConfig(
    system_instruction=AGENT_INSTRUCTION, tools=TOOLS, temperature=0.2, thinking_config=_thinking,
    # The agent graph runs the tools itself, so the SDK must not call anything automatically.
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


class LLMError(Exception):
    pass


class Answer(BaseModel):
    answer: str
    found: bool
    sources: list[str]


class TicketSummary(BaseModel):
    summary: str
    category: str


@dataclass
class Turn:
    """One model turn: the reply (if any), the tools it called, and the turn to send back next time."""
    content: types.Content  # the whole turn, thought signatures included: Gemini needs it back
    calls: list[types.FunctionCall]
    answer: Answer


@contextmanager
def _gemini_errors():
    try:
        yield
    except errors.APIError as e:
        raise LLMError(f"Gemini API error {e.code}: {e.message}") from e
    except httpx.HTTPError as e:
        raise LLMError(f"Could not reach Gemini: {type(e).__name__}") from e


def _generate(prompt: str, schema: type[BaseModel]) -> BaseModel:
    """One-shot call with JSON output validated against a Pydantic schema."""
    with _gemini_errors():
        response = _client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=schema, temperature=0.2,
                thinking_config=_thinking,
            ),
        )
    if not isinstance(response.parsed, schema):
        raise LLMError("Gemini returned an unexpected response")
    return response.parsed


def parse_answer_stream(pieces: Iterable[str]) -> Iterator[str | Answer]:
    """Turn raw model text pieces into displayable text pieces, then yield one final Answer.

    Text is held back while it could still be NOT_FOUND, and the trailing SOURCES line is
    never shown, so the customer only ever sees the reply itself.
    """
    text, sent = "", 0
    for piece in pieces:
        text += piece
        if NOT_FOUND.startswith(text.strip()[:len(NOT_FOUND)].upper()):
            continue  # could still turn out to be NOT_FOUND
        sources_at = SOURCES_LINE.search(text)
        end = sources_at.start() if sources_at else len(text) - HOLD_BACK
        if end > sent:
            yield text[sent:end]
            sent = end

    sources_at = SOURCES_LINE.search(text)
    answer_end = sources_at.start() if sources_at else len(text)
    answer = text[:answer_end].strip()
    if not answer or answer.upper().startswith(NOT_FOUND):
        yield Answer(answer="", found=False, sources=[])
        return
    if answer_end > sent:
        yield text[sent:answer_end]
    sources = [s.strip(" []*`'\"\n") for s in text[sources_at.end():].split(",")] if sources_at else []
    yield Answer(answer=answer, found=True, sources=[s for s in sources if s])


def _transcript(messages: list[dict]) -> str:
    return "\n".join(f"{SPEAKERS[m['role']]}: {m['content']}" for m in messages)


def user_turn(message: str, history: list[dict]) -> types.Content:
    """The customer's message, with the conversation so far as the agent's memory."""
    text = f"CONVERSATION:\n{_transcript(history) or '(this is the first message)'}\n\nMESSAGE: {message}"
    return types.Content(role="user", parts=[types.Part(text=text)])


def tool_results(calls: list[types.FunctionCall], results: list[dict]) -> types.Content:
    """Tool outputs for the model, matched to its calls by id. The API takes them as a user turn."""
    return types.Content(role="user", parts=[
        types.Part(function_response=types.FunctionResponse(id=c.id, name=c.name, response=r))
        for c, r in zip(calls, results)
    ])


def stream_turn(contents: list[types.Content]) -> Iterator[str | Turn]:
    """One model turn with the tools available: yield reply text pieces as they arrive, then the Turn."""
    parts: list[types.Part] = []

    def text_pieces(stream):
        for chunk in stream:
            candidate = chunk.candidates[0] if chunk.candidates else None
            new = (candidate.content.parts or []) if candidate and candidate.content else []
            parts.extend(new)
            yield "".join(p.text for p in new if p.text and not p.thought)

    with _gemini_errors():
        stream = _client.models.generate_content_stream(model=MODEL, contents=contents, config=_AGENT_CONFIG)
        for item in parse_answer_stream(text_pieces(stream)):
            if isinstance(item, Answer):
                calls = [p.function_call for p in parts if p.function_call]
                yield Turn(types.Content(role="model", parts=parts), calls, item)
            else:
                yield item


def summarize_ticket(transcript: list[dict]) -> TicketSummary:
    prompt = (f"Summarize this customer support conversation as a support ticket.\n"
              f"summary: one sentence describing the customer's issue (max 25 words).\n"
              f"category: exactly one of {', '.join(CATEGORIES)}.\n\nCONVERSATION:\n{_transcript(transcript)}")
    result = _generate(prompt, TicketSummary)
    if result.category not in CATEGORIES:
        result.category = "general"
    return result
