import { Fragment, useEffect, useRef, useState } from 'react'
import { formatDate, postJSON, request } from './api.js'

const STATUSES = ['Open', 'In Progress', 'Resolved']
const STATUS_CLASS = { Open: 'pink', 'In Progress': 'canary', Resolved: 'paper' }

export default function Tickets({ focusId }) {
  const [tickets, setTickets] = useState(null)
  const [filter, setFilter] = useState('All')
  const [openId, setOpenId] = useState(null)
  // Transcripts by ticket id, so a slow response for one ticket can't land under another.
  const [details, setDetails] = useState({})
  const [error, setError] = useState('')
  const focusRow = useRef(null)
  const scrolledToFocus = useRef(false)

  const load = () => request('/tickets').then(setTickets).catch((e) => { setTickets([]); setError(e.message) })
  useEffect(() => { load() }, [])

  // Arriving from a ticket stub in the chat: open that ticket and bring it into view once.
  useEffect(() => { if (focusId) toggle(focusId) }, [focusId])
  useEffect(() => {
    if (focusRow.current && !scrolledToFocus.current) {
      focusRow.current.scrollIntoView({ block: 'center' })
      scrolledToFocus.current = true
    }
  }, [tickets])

  async function toggle(id) {
    if (openId === id) return setOpenId(null)
    setOpenId(id)
    try {
      const detail = await request(`/tickets/${id}`)
      setDetails((d) => ({ ...d, [id]: detail }))
    } catch (e) { setError(e.message) }
  }

  async function setStatus(id, status) {
    setError('')
    setTickets((ts) => ts.map((t) => (t.id === id ? { ...t, status } : t))) // update now, confirm below
    try { await postJSON(`/tickets/${id}`, { status }, 'PATCH') } catch (e) { setError(e.message); load() }
  }

  const count = (s) => tickets?.filter((t) => s === 'All' || t.status === s).length ?? 0
  const shown = tickets?.filter((t) => filter === 'All' || t.status === filter) ?? []

  return (
    <section className="tickets">
      <div className="page-head">
        <h1 className="display">Tickets</h1>
        <button className="btn ghost" onClick={load}>Refresh</button>
      </div>
      <p className="lede">Conversations the agent handed to a person, with the full chat attached.</p>

      <div className="filters" role="group" aria-label="Filter by status">
        {['All', ...STATUSES].map((s) => (
          <button key={s} aria-pressed={filter === s} onClick={() => setFilter(s)}>
            {s} <span className="count">{count(s)}</span>
          </button>
        ))}
      </div>
      {error && <p className="error" role="alert">{error}</p>}

      {tickets === null && <p className="muted">Loading…</p>}
      {tickets && shown.length === 0 && (
        <p className="empty">
          {tickets.length === 0
            ? 'No tickets yet. Ask the agent something it can’t answer, or ask for a person, to create one.'
            : `No ${filter.toLowerCase()} tickets.`}
        </p>
      )}

      {shown.length > 0 && (
        <div className="table-wrap">
        <table className="ticket-table">
          <thead>
            <tr>
              <th>Ticket</th><th>Issue</th><th className="hide-sm">Customer</th>
              <th className="hide-sm">Category</th><th>Status</th><th className="hide-sm">Created</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((t) => (
              <Fragment key={t.id}>
                <tr className={`ticket-row ${STATUS_CLASS[t.status]} ${openId === t.id ? 'open' : ''}`}
                  ref={t.id === focusId ? focusRow : null}>
                  <td className="ticket-no">#{t.id}</td>
                  <td>
                    <button className="link" aria-expanded={openId === t.id} onClick={() => toggle(t.id)}>
                      {t.summary}
                    </button>
                    <span className="muted small block">{t.reason}</span>
                  </td>
                  <td className="hide-sm">{t.customer_name}<span className="muted small block">{t.customer_email}</span></td>
                  <td className="cap hide-sm">{t.category}</td>
                  <td>
                    <select className={`status ${STATUS_CLASS[t.status]}`} value={t.status}
                      aria-label={`Status of ticket ${t.id}`} onChange={(e) => setStatus(t.id, e.target.value)}>
                      {STATUSES.map((s) => <option key={s}>{s}</option>)}
                    </select>
                  </td>
                  <td className="small hide-sm nowrap">{formatDate(t.created_at)}</td>
                </tr>
                {openId === t.id && (
                  <tr className="transcript-row">
                    <td colSpan="6">
                      {!details[t.id] ? <p className="muted">Loading conversation…</p> : (
                        <ol className="transcript" aria-label={`Conversation for ticket ${t.id}`}>
                          {details[t.id].messages.map((m) => (
                            <li key={m.id} className={m.role}>
                              <span className="who">{m.role === 'user' ? t.customer_name : 'Agent'}</span>
                              <p>{m.content}</p>
                            </li>
                          ))}
                        </ol>
                      )}
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
        </div>
      )}
    </section>
  )
}
