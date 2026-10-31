# AI Usage

**AI tool:** Claude Code (Anthropic), CLI, with three add-ons:
- **Ponytail** (plugin, "full" mode): pushes each change toward the smallest version that works. It shaped choices like stdlib `sqlite3` instead of an ORM, Chroma's built-in embedding model instead of installing PyTorch, plain CSS, and no router.
- **Context7** (MCP server): fetched current library docs before code was written against them, so the code follows today's APIs rather than the model's memory. Used for google-genai (JSON output, streaming, thinking level, function calling), ChromaDB (cosine search), FastAPI (streaming responses) and LangGraph (routing, streaming from nodes).
- **frontend-design** (skill): guided the redesign. The design was planned first (palette, fonts, layout) and checked against generic AI-looking defaults, which led to the carbon-copy service-desk look: indigo ink, canary and pink ticket colours, a perforated ticket stub. The result was checked in browser screenshots.

**How it was used:**
- Planned the architecture and build order from the assignment document.
- Generated the FastAPI backend (routes, SQLite schema, chunking, Chroma integration, Gemini calls) and the React components (Knowledge Base, Chat, Tickets).
- Designed the escalation rules with me and wrote the unit tests for chunking and escalation.
- Ran a smoke test against the API. It caught a real chunking bug (overlap starting mid-word).
- Debugged "the bot can't find anything": broad questions scored below the first similarity threshold (0.25) and were escalated without asking the LLM. Lowered it to 0.10 based on measured scores, and made source citations come from what the LLM reports it used.
- Added streamed responses (`/chat/stream`, JSON Lines). Measured time-to-first-token and set Gemini's thinking level to `low` (~10s → ~2s).
- Fixed "the bot doesn't remember the conversation": a multi-turn test showed it treated every message as a standalone docs question ("yes" to a ticket offer, "thanks", "what's my name?" all failed). It now uses the last 20 messages as memory, can accept a ticket offer, and handles small talk, while company facts still come only from the docs.
- Moved the chat flow onto LangGraph (`backend/agent.py`): the existing steps became graph nodes with conditional routing, plus a test that drives every route with Gemini faked. Re-ran the multi-turn memory scenarios against real Gemini to confirm behaviour didn't change.
- Turned the agent into a tool-calling agent: Gemini calls `search_help_docs` and `create_support_ticket`, LangGraph runs the tools and loops, and the keyword rules stay in front as a guardrail. A live probe caught that the API rejects the `tool` role for function results (they go back as `user`) before it reached the app. Re-ran 12 multi-turn scenarios against real Gemini and checked the live search status in the browser.
- Audited the frontend in a real browser (Playwright + Edge) and fixed six bugs it reproduced: a raw regex error for emails like `name@example`, long unbroken text overflowing the chat, a reply lost when switching tabs mid-answer, an old reply leaking into a new conversation, the saved conversation being forgotten when the backend was briefly unreachable, and a slow ticket transcript appearing under a different ticket.
- Wrote the Dockerfile and `compose.yaml`: Node builds the React app, then a Python image serves it from FastAPI, so the whole app runs on one port, and Compose passes the key from `.env` at startup. A `.dockerignore` keeps `.env` and local data out of the image, and the Python dependencies are pinned to the tested versions. Built and ran the image: the unit tests pass inside it, upload, chat, streaming and escalation work, and the data survives replacing the container. For Render, made the container listen on the `PORT` it provides, and ran it under the free plan's limits (512 MB RAM, 0.1 CPU) to check that uploads and chat fit.
- Drafted the README and this file.


