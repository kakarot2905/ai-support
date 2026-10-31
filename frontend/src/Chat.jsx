import { useEffect, useRef, useState } from 'react'
import { postJSON, request, streamJSONLines } from './api.js'

const STORAGE_KEY = 'conversationId'

// Starter questions that exercise each path: KB answers, a billing escalation, a human handoff.
const STARTERS = [
  'How long do I have to return an item?',
  'Do you ship internationally?',
  'I was charged twice for my last order',
  "I'd like to talk to a person",
]

function savedConversationId() {
  try { return localStorage.getItem(STORAGE_KEY) } catch { return null }
}

function rememberConversation(id) {
  try { id ? localStorage.setItem(STORAGE_KEY, id) : localStorage.removeItem(STORAGE_KEY) } catch { /* storage blocked */ }
}

export default function Chat({ active, onOpenTicket }) {
  const [conversation, setConversation] = useState(null)
  const [messages, setMessages] = useState([])
  const [loadingHistory, setLoadingHistory] = useState(!!savedConversationId())
  const [restoreError, setRestoreError] = useState('')
  const [busy, setBusy] = useState(false) // a reply is in progress

  // Restore the conversation persisted on the server (survives reloads and restarts).
  // Only a 404 means it's gone; any other failure (e.g. backend not started yet) keeps it for a retry.
  function restore() {
    const id = savedConversationId()
    if (!id) return
    setLoadingHistory(true)
    setRestoreError('')
    request(`/conversations/${id}`)
      .then((c) => { setConversation(c); setMessages(c.messages) })
      .catch((e) => (e.status === 404 ? rememberConversation(null) : setRestoreError(e.message)))
      .finally(() => setLoadingHistory(false))
  }
  useEffect(() => { restore() }, [])

  function start(conv) {
    rememberConversation(conv.id)
    setConversation(conv)
    setMessages([])
  }

  function reset() {
    rememberConversation(null)
    setConversation(null)
    setMessages([])
  }

  if (loadingHistory) return <p className="muted">Loading your conversation…</p>
  if (restoreError) {
    return (
      <section className="start">
        <p className="error" role="alert">Couldn&apos;t load your conversation. {restoreError}</p>
        <button className="btn ghost" onClick={restore}>Try again</button>
      </section>
    )
  }
  if (!conversation) return <StartForm onStart={start} />

  return (
    <section className="chat">
      <header className="chat-bar">
        <div>
          <span className="chat-who">{conversation.customer_name}</span>
          <span className="muted"> {conversation.customer_email}</span>
        </div>
        <div className="chat-bar-actions">
          {conversation.status === 'escalated' && <span className="pill pink">With our team</span>}
          {/* Disabled mid-reply: the reply still streaming would otherwise land in the new conversation. */}
          <button className="btn ghost" onClick={reset} disabled={busy}>New conversation</button>
        </div>
      </header>
      <ChatWindow
        active={active}
        busy={busy}
        setBusy={setBusy}
        conversation={conversation}
        messages={messages}
        setMessages={setMessages}
        onEscalated={() => setConversation((c) => ({ ...c, status: 'escalated' }))}
        onOpenTicket={onOpenTicket}
      />
    </section>
  )
}

function StartForm({ onStart }) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(e) {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      onStart(await postJSON('/conversations', { name, email }))
    } catch (err) {
      setError(err.message)
      setBusy(false)
    }
  }

  return (
    <section className="start">
      <h1 className="display">How can we help?</h1>
      <p className="lede">
        Ask about orders, shipping or returns. If I can&apos;t answer, I&apos;ll open a ticket and a person
        will get back to you by email.
      </p>
      <form className="sheet start-form" onSubmit={submit}>
        <label>
          Your name
          <input value={name} onChange={(e) => setName(e.target.value)} autoComplete="name" required />
        </label>
        <label>
          Email for replies
          {/* Same rule as the server, so the browser catches "name@example" with a readable message. */}
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" required
            pattern="[^@\s]+@[^@\s]+\.[^@\s]+" title="Enter an email like name@example.com" />
        </label>
        {error && <p className="error" role="alert">{error}</p>}
        <button className="btn" disabled={busy}>{busy ? 'Starting…' : 'Start chatting'}</button>
      </form>
    </section>
  )
}

function ChatWindow({ active, busy, setBusy, conversation, messages, setMessages, onEscalated, onOpenTicket }) {
  const [input, setInput] = useState('')
  const [error, setError] = useState('')
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  // Also on returning to the tab: replies that arrived while it was hidden are below the fold.
  useEffect(() => { if (active) bottomRef.current?.scrollIntoView({ block: 'end' }) }, [messages, busy, active])

  async function send(text) {
    text = text.trim()
    if (!text) return setError('Type a question first.')
    setError('')
    setInput('')
    // Add the user message plus an empty agent message that fills in as tokens stream.
    setMessages((m) => [...m, { role: 'user', content: text }, { role: 'agent', content: '', streaming: true }])
    setBusy(true)
    try {
      await streamJSONLines('/chat/stream', { conversation_id: conversation.id, message: text }, (event) => {
        if (event.type === 'error') throw new Error(event.detail)
        setMessages((m) => {
          const last = m.at(-1)
          const updated =
            event.type === 'token' ? { ...last, content: last.content + event.text }
            // "status" says what the agent is doing, e.g. which search it's running.
            : event.type === 'status' ? { ...last, status: event.text }
            // "done" carries the final saved reply, so the message always matches what's stored.
            : {
                role: 'agent', content: event.reply, sources: event.sources, ticket_id: event.ticket_id,
                offer_ticket: event.offer_ticket, fresh: true,
              }
          return [...m.slice(0, -1), updated]
        })
        if (event.escalated) onEscalated()
      })
    } catch (err) {
      setMessages((m) => (m.at(-1)?.streaming ? m.slice(0, -1) : m)) // drop the unfinished message
      setError(err.message)
    } finally {
      setBusy(false)
      inputRef.current?.focus()
    }
  }

  async function createTicket() {
    setError('')
    setBusy(true)
    try {
      const ticket = await postJSON('/tickets', { conversation_id: conversation.id })
      const { messages: saved } = await request(`/conversations/${conversation.id}`)
      setMessages(saved.map((m) => (m.ticket_id === ticket.id ? { ...m, fresh: true } : m)))
      onEscalated()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const escalated = conversation.status === 'escalated'

  return (
    <>
      <div className="thread" aria-live="polite">
        {messages.length === 0 && (
          <div className="welcome">
            <h1 className="display">Hi {conversation.customer_name.split(' ')[0]}, what&apos;s your question?</h1>
            <p className="muted">Try one of these, or type your own below.</p>
            <div className="starters">
              {STARTERS.map((q) => (
                <button key={q} className="starter" onClick={() => send(q)} disabled={busy}>{q}</button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          m.role === 'user'
            ? <div key={i} className="msg user"><p className="bubble">{m.content}</p></div>
            : (
              <div key={i} className="msg agent">
                <span className="avatar" aria-hidden="true">?</span>
                <div className="agent-body">
                  {m.streaming && !m.content
                    ? (
                      <span className="working" role="status">
                        <span className="dots" aria-hidden="true"><i /><i /><i /></span>
                        <span className={m.status ? 'muted small' : 'visually-hidden'}>{m.status || 'Writing an answer'}</span>
                      </span>
                    )
                    : <p className={m.streaming ? 'agent-text streaming' : 'agent-text'}>{m.content}</p>}
                  {m.sources?.length > 0 && (
                    <ul className="sources" aria-label="Sources">
                      {m.sources.map((s) => <li key={s}>{s}</li>)}
                    </ul>
                  )}
                  {m.ticket_id && (
                    <TicketStub id={m.ticket_id} fresh={m.fresh}
                      onOpen={() => onOpenTicket(m.ticket_id)} />
                  )}
                  {m.offer_ticket && !escalated && (
                    <button className="btn ghost" onClick={createTicket} disabled={busy}>Create a support ticket</button>
                  )}
                </div>
              </div>
            )
        ))}
        {busy && !messages.at(-1)?.streaming && (
          <div className="msg agent">
            <span className="avatar" aria-hidden="true">?</span>
            <span className="dots" role="status" aria-label="Working"><i /><i /><i /></span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && <p className="error composer-error" role="alert">{error}</p>}
      <form className="composer" onSubmit={(e) => { e.preventDefault(); send(input) }}>
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={escalated ? 'Add more details for our team…' : 'Ask about orders, shipping, returns…'}
          aria-label="Your message"
          maxLength={2000}
          autoFocus
        />
        <button className="btn" disabled={busy || !input.trim()}>Send</button>
      </form>
    </>
  )
}

// The escalation moment: a perforated ticket stub that "prints" into the thread when it's new.
function TicketStub({ id, fresh, onOpen }) {
  return (
    <div className={`stub ${fresh ? 'printing' : ''}`}>
      <div className="stub-number">
        <span>Ticket</span>
        <strong>#{id}</strong>
      </div>
      <div className="stub-body">
        <p>Your ticket is open. Mention <b>#{id}</b> if you contact us about this again.</p>
        <button className="btn small" onClick={onOpen}>View ticket</button>
      </div>
    </div>
  )
}
