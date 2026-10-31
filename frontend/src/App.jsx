import { useState } from 'react'
import KnowledgeBase from './KnowledgeBase.jsx'
import Chat from './Chat.jsx'
import Tickets from './Tickets.jsx'

const icon = (d) => (
  <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" fill="none" stroke="currentColor"
    strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{d}</svg>
)

const TABS = [
  { key: 'chat', label: 'Chat', icon: icon(<path d="M4 5h16v11H9l-5 4z" />) },
  { key: 'kb', label: 'Knowledge base', icon: icon(<><path d="M6 3h9l4 4v14H6z" /><path d="M14 3v5h5M9 13h7M9 17h5" /></>) },
  {
    key: 'tickets', label: 'Tickets',
    icon: icon(<path d="M3 7h18v3a2 2 0 0 0 0 4v3H3v-3a2 2 0 0 0 0-4zM14 7v10" />),
  },
]

export default function App() {
  const [tab, setTab] = useState('chat')
  const [focusTicketId, setFocusTicketId] = useState(null)

  function openTicket(id) {
    setFocusTicketId(id)
    setTab('tickets')
  }

  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">?</span>
          <span>Support Agent</span>
        </div>
        <nav aria-label="Main">
          {TABS.map((t) => (
            <button
              key={t.key}
              className="nav-item"
              aria-current={tab === t.key ? 'page' : undefined}
              onClick={() => { setFocusTicketId(null); setTab(t.key) }}
            >
              {t.icon}
              <span>{t.label}</span>
            </button>
          ))}
        </nav>
        <p className="rail-note">Answers come only from your uploaded documents. Anything else goes to a person.</p>
      </aside>
      <main className="page">
        {/* Chat stays mounted (just hidden) so a reply that's streaming survives a tab switch.
            The other tabs remount, which refreshes their lists every time they're opened. */}
        <div hidden={tab !== 'chat'}>
          <Chat active={tab === 'chat'} onOpenTicket={openTicket} />
        </div>
        {tab === 'kb' && <KnowledgeBase />}
        {tab === 'tickets' && <Tickets focusId={focusTicketId} />}
      </main>
    </div>
  )
}
