"""Unit tests for chunking, escalation rules, answer-stream parsing and the agent graph. Run: cd backend && pytest"""
import os

os.environ.setdefault("GEMINI_API_KEY", "test-key")  # llm.py needs a key to import; no API calls are made

from escalation import check_triggers
from kb import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text
from llm import Answer, parse_answer_stream


def run_stream(pieces):
    *texts, final = parse_answer_stream(pieces)
    assert isinstance(final, Answer)
    return "".join(texts), final


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= CHUNK_SIZE for c in chunks)
    # consecutive chunks share overlapping text
    assert chunks[0][-CHUNK_OVERLAP // 2:].split()[-1] in chunks[1]
    # no words cut in half: every token is a full "wordN"
    assert all(tok.startswith("word") for c in chunks for tok in c.split())


def test_short_and_empty_text():
    assert chunk_text("Returns accepted within 30 days.") == ["Returns accepted within 30 days."]
    assert chunk_text("   ") == []


def test_explicit_human_request_escalates():
    assert check_triggers("Can I talk to a human please?")[1] == "You asked to speak with a person"
    assert check_triggers("That didn't help at all") is not None


def test_billing_from_assignment_example_escalates():
    assert check_triggers("I was charged twice for my last order and need a refund now.")[0] == "billing"


def test_policy_questions_do_not_escalate():
    assert check_triggers("What is your refund policy?") is None
    assert check_triggers("How long do I have to return an item?") is None


def test_stream_hides_sources_line_even_when_split_across_pieces():
    text, final = run_stream(["You can return ", "items within 30 days.", "\nSOU", "RCES: Return_Policy.txt, FAQ.txt"])
    assert text == "You can return items within 30 days."
    assert final.found and final.sources == ["Return_Policy.txt", "FAQ.txt"]


def test_stream_not_found_is_never_shown():
    text, final = run_stream(["NOT_", "FOUND"])
    assert text == "" and not final.found


def test_stream_without_sources_line_still_answers():
    text, final = run_stream(["Standard shipping takes ", "3 to 5 business days."])
    assert text == "Standard shipping takes 3 to 5 business days."
    assert final.found and final.sources == []


def test_stream_text_that_starts_like_not_found_is_still_shown():
    text, final = run_stream(["No", "t yet: orders ship ", "within 2 days."])
    assert text == "Not yet: orders ship within 2 days."
    assert final.found


def test_agent_graph_tools_routes_and_memory(monkeypatch, tmp_path):
    """Runs the LangGraph agent with Gemini scripted and search faked, on a throwaway database."""
    import agent, db, kb, llm
    from google.genai import types
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init()
    monkeypatch.setattr(kb, "search", lambda query: [{"text": "Returns within 30 days.", "source": "Policy.txt",
                                                       "similarity": 0.8}])
    conv_id = db.execute("INSERT INTO conversations (customer_name, customer_email, created_at) "
                         "VALUES ('Asha', 'asha@example.com', 'now')")
    conv = db.query_one("SELECT * FROM conversations WHERE id = ?", (conv_id,))
    model_inputs = []  # what each model turn received

    def call(name, **args):
        return types.FunctionCall(id=f"call-{name}", name=name, args=args)

    def turn(*pieces, calls=(), reply=""):
        """A scripted model turn: streamed text pieces, then the Turn itself."""
        return [*pieces, llm.Turn(types.Content(role="model", parts=[]), list(calls),
                                  Answer(answer=reply, found=bool(reply), sources=["Policy.txt", "Made_up.txt"]))]

    def run(message, *turns):
        script = iter(turns)
        def fake_turn(contents):
            model_inputs.append(list(contents))
            return iter(next(script))
        monkeypatch.setattr(llm, "stream_turn", fake_turn)
        events = list(agent.chat_events(conv, message))
        return events, events[-1]

    # Search, then answer: status + tokens stream, the passages go back to the model,
    # and only files the search actually returned are cited.
    events, done = run("how long to return?",
                       turn(calls=[call(llm.SEARCH_TOOL, query="return window")]),
                       turn("Within ", "30 days.", reply="Within 30 days."))
    assert [e["text"] for e in events if e["type"] == "token"] == ["Within ", "30 days."]
    assert any(e["type"] == "status" and "return window" in e["text"] for e in events)
    assert done["reply"] == "Within 30 days." and done["sources"] == ["Policy.txt"] and not done["ticket_id"]
    tool_reply = model_inputs[-1][-1].parts[0].function_response
    assert tool_reply.name == llm.SEARCH_TOOL and tool_reply.response["passages"][0]["source"] == "Policy.txt"

    # Not found: offer a ticket. Memory: the first exchange is in the prompt.
    _, done = run("what do you sell?", turn())
    assert done["offer_ticket"] and not done["ticket_id"]
    assert "Customer: how long to return?" in model_inputs[-1][0].parts[0].text

    # The model opens the ticket itself and writes its summary (no separate summary call).
    monkeypatch.setattr(llm, "summarize_ticket", lambda transcript: (_ for _ in ()).throw(AssertionError("unused")))
    _, done = run("yes please", turn(calls=[call(llm.TICKET_TOOL, summary="Wants the product list", category="general")]))
    ticket = db.query_one("SELECT * FROM tickets WHERE id = ?", (done["ticket_id"],))
    assert ticket["summary"] == "Wants the product list" and ticket["category"] == "general"

    # Keyword guardrail: escalates before the model is called, reusing the open ticket.
    turns_so_far = len(model_inputs)
    _, done = run("I was charged twice")
    assert len(model_inputs) == turns_so_far and "already with our team" in done["reply"]

    # A model that keeps searching is stopped after MAX_SEARCHES rounds.
    searching = turn(calls=[call(llm.SEARCH_TOOL, query="anything")])
    _, done = run("loop forever", *[searching] * (agent.MAX_SEARCHES + 1))
    assert "couldn't find that" in done["reply"] and len(model_inputs) == turns_so_far + agent.MAX_SEARCHES + 1
