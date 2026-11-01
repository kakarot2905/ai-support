"""The support agent: a LangGraph state graph where Gemini decides which tools to use.

    START → remember → check_rules ─┬─ keyword rule matched ──────────────────────→ escalate ──→ END
                                    └→ agent ─┬─ calls search_help_docs ──→ search ─→ agent (loop)
                                              ├─ calls create_support_ticket ──────→ escalate ──→ END
                                              ├─ replies NOT_FOUND ────────────────→ not_found → END
                                              └─ replies ──────────────────────────→ answer ───→ END

The keyword rules stay in front of the model as a guardrail, so billing, account and "talk to a
person" messages always escalate. Everything else is the model's call: when and what to search,
when to open a ticket (it writes the summary and category itself), and what to reply. `agent`
streams the reply text with LangGraph's stream writer; `search` also streams a status line.
Memory is the conversation in SQLite (the same messages the UI and tickets show), loaded by `remember`.
"""
import logging
import operator
from collections.abc import Iterator
from typing import Annotated, Literal, TypedDict

from fastapi import HTTPException
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

import db
import escalation
import kb
import llm
import mailer

log = logging.getLogger("support")

# Conversation memory: how many earlier messages the LLM sees. Bounded so a very long chat
# can't grow the prompt (and its cost and latency) without limit.
HISTORY_MESSAGES = 20
MAX_SEARCHES = 2  # search rounds per message; a model still searching after that gets "not found"
TICKET_REQUESTED = "You asked for a support ticket"


class ChatState(TypedDict, total=False):
    conv: dict                                   # the conversation row (id, customer name/email, status)
    message: str                                 # the customer's latest message
    history: list[dict]                          # earlier messages: the agent's memory
    trigger: tuple[str | None, str] | None       # (category, reason) if a keyword rule matched
    contents: Annotated[list, operator.add]      # this message's exchange with the model: prompt, turns, tool results
    turn: llm.Turn                               # the model's latest turn: reply, tool calls
    text: str                                    # reply text streamed so far
    sources: Annotated[list[str], operator.add]  # files the searches returned
    searches: int                                # search rounds so far
    done: dict                                   # the final chat event sent to the customer


# ---------- Nodes ----------

def remember(state: ChatState) -> ChatState:
    """Load the conversation so far (memory), then add the new message to it."""
    conv_id = state["conv"]["id"]
    history = db.get_messages(conv_id)[-HISTORY_MESSAGES:]
    db.add_message(conv_id, "user", state["message"])
    return {"history": history, "contents": [llm.user_turn(state["message"], history)]}


def check_rules(state: ChatState) -> ChatState:
    """Explicit keyword rules: asked for a person, billing, account access, complaints."""
    return {"trigger": escalation.check_triggers(state["message"])}


def agent(state: ChatState) -> ChatState:
    """One model turn: Gemini replies (streamed to the customer as it writes) or calls a tool."""
    write = get_stream_writer()
    before, started = state.get("text", ""), False
    for item in llm.stream_turn(state["contents"]):
        if isinstance(item, llm.Turn):
            turn = item
            continue
        if before and not started:  # more text after a tool call starts a new paragraph
            write({"type": "token", "text": "\n\n"})
        started = True
        write({"type": "token", "text": item})
    text = "\n\n".join(t for t in (before, turn.answer.answer) if t)
    return {"turn": turn, "contents": [turn.content], "text": text}


def search(state: ChatState) -> ChatState:
    """Run the model's search_help_docs calls and hand the passages back to it."""
    write = get_stream_writer()
    calls, results, found = state["turn"].calls, [], []
    for call in calls:
        if call.name != llm.SEARCH_TOOL:  # every call needs an answer, even one we don't run here
            results.append({"error": f"{call.name} can't run together with a search"})
            continue
        query = (call.args or {}).get("query") or state["message"]
        write({"type": "status", "text": f"Searching the help docs: {query}"})
        chunks = kb.search(query)
        found += [c["source"] for c in chunks]
        results.append({"passages": [{"source": c["source"], "text": c["text"]} for c in chunks]})
    return {"contents": [llm.tool_results(calls, results)], "sources": found,
            "searches": state.get("searches", 0) + 1}


def hand_off(state: ChatState) -> ChatState:
    """Create (or point to) the support ticket: a keyword rule matched, or the model called the ticket tool."""
    if state.get("trigger"):
        category, reason = state["trigger"]
        return {"done": escalate(state["conv"], reason, category)}
    args = next(c.args or {} for c in state["turn"].calls if c.name == llm.TICKET_TOOL)
    category = args.get("category") if args.get("category") in llm.CATEGORIES else None
    return {"done": escalate(state["conv"], TICKET_REQUESTED, category, args.get("summary"))}


def not_found(state: ChatState) -> ChatState:
    """Nothing in the docs or the conversation answers it: say so, and offer a ticket."""
    conv_id = state["conv"]["id"]
    ticket = open_ticket(conv_id)
    if ticket:
        return {"done": respond(conv_id, "I couldn't find that in our help docs, but our team can see this "
                                f"conversation on your ticket #{ticket['id']}.")}
    return {"done": respond(conv_id, "I couldn't find that in our help docs. "
                            "Would you like me to create a support ticket so our team can help?", offer_ticket=True)}


def answer(state: ChatState) -> ChatState:
    """Send the reply, citing the files the model says it used, but only ones its searches returned."""
    found = set(state.get("sources", []))
    sources = [s for s in dict.fromkeys(state["turn"].answer.sources) if s in found]
    return {"done": respond(state["conv"]["id"], state["text"], sources)}


# ---------- Routing ----------

def after_rules(state: ChatState) -> Literal["escalate", "agent"]:
    return "escalate" if state["trigger"] else "agent"


def after_agent(state: ChatState) -> Literal["search", "escalate", "not_found", "answer"]:
    turn = state["turn"]
    called = {c.name for c in turn.calls}
    if llm.TICKET_TOOL in called:
        return "escalate"
    if llm.SEARCH_TOOL in called:
        return "search" if state.get("searches", 0) < MAX_SEARCHES else "not_found"
    return "answer" if turn.answer.found else "not_found"


graph = (
    StateGraph(ChatState)
    .add_node("remember", remember)
    .add_node("check_rules", check_rules)
    .add_node("agent", agent)
    .add_node("search", search)
    .add_node("escalate", hand_off)
    .add_node("not_found", not_found)
    .add_node("answer", answer)
    .add_edge(START, "remember")
    .add_edge("remember", "check_rules")
    .add_conditional_edges("check_rules", after_rules)
    .add_conditional_edges("agent", after_agent)
    .add_edge("search", "agent")
    .add_edge("escalate", END)
    .add_edge("not_found", END)
    .add_edge("answer", END)
    .compile()
)


def chat_events(conv: dict, message: str) -> Iterator[dict]:
    """Run the graph for one message: "token" and "status" events while it works, then "done"."""
    try:
        for part in graph.stream({"conv": conv, "message": message},
                                 stream_mode=["custom", "updates"], version="v2"):
            if part["type"] == "custom":
                yield part["data"]
            else:
                for update in part["data"].values():
                    if update and "done" in update:
                        yield update["done"]
    except llm.LLMError as e:
        log.error("LLM failure: %s", e)
        raise HTTPException(503, f"The AI service is unavailable right now ({e}). Please try again.")


# ---------- Replies and tickets (also used by the /tickets route) ----------

def respond(conv_id: int, text: str, sources: list[str] | None = None,
            ticket_id: int | None = None, offer_ticket: bool = False) -> dict:
    """Save the agent's reply and build the final ("done") chat event."""
    db.add_message(conv_id, "agent", text, sources, ticket_id)
    return {"type": "done", "reply": text, "sources": sources or [], "ticket_id": ticket_id,
            "escalated": ticket_id is not None, "offer_ticket": offer_ticket}


def open_ticket(conversation_id: int) -> dict | None:
    """The conversation's unresolved ticket, if it has one."""
    return db.query_one(
        "SELECT * FROM tickets WHERE conversation_id = ? AND status != 'Resolved' ORDER BY id DESC",
        (conversation_id,),
    )


def create_ticket(conv: dict, reason: str, category: str | None = None,
                  summary: str | None = None) -> tuple[dict, bool]:
    """Create a ticket for the conversation, or return its existing unresolved one.

    Returns (ticket, created). Never fails because of the LLM: falls back to the last message.
    """
    existing = open_ticket(conv["id"])
    if existing:
        return existing, False
    if not conv["customer_email"]:
        raise HTTPException(422, "Customer contact email is required to create a ticket.")

    transcript = db.get_messages(conv["id"])
    if not summary or not category:
        try:
            s = llm.summarize_ticket(transcript)
            summary, category = summary or s.summary, category or s.category
        except llm.LLMError as e:
            log.warning("Ticket summary fell back to last message: %s", e)
            last_user = next((m["content"] for m in reversed(transcript) if m["role"] == "user"), reason)
            summary, category = summary or last_user[:200], category or "general"

    ticket_id = db.execute(
        "INSERT INTO tickets (conversation_id, summary, customer_name, customer_email, category, reason, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (conv["id"], summary, conv["customer_name"], conv["customer_email"], category, reason, db.now()),
    )
    db.execute("UPDATE conversations SET status = 'escalated' WHERE id = ?", (conv["id"],))
    ticket = db.query_one("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    mailer.send_ticket_email(ticket)
    return ticket, True


def escalate(conv: dict, reason: str, category: str | None = None, summary: str | None = None) -> dict:
    """Hand the conversation to a person: create (or reuse) its ticket and tell the customer."""
    ticket, created = create_ticket(conv, reason, category, summary)
    if created:
        text = (f"{reason}, so I've created support ticket #{ticket['id']}. "
                f"Our team will follow up with you at {ticket['customer_email']}.")
    else:
        text = (f"Your support ticket #{ticket['id']} is already with our team. "
                f"They'll follow up with you at {ticket['customer_email']}.")
    return respond(conv["id"], text, ticket_id=ticket["id"])
