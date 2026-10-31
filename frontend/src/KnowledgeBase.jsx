import { useEffect, useRef, useState } from 'react'
import { formatDate, request } from './api.js'

const ACCEPTED = ['.pdf', '.txt', '.docx']
const STATUS_PILL = { Processed: 'ink', Processing: 'canary', Failed: 'pink' }

export default function KnowledgeBase() {
  const [docs, setDocs] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const fileInput = useRef(null)

  const load = () => request('/kb/documents').then(setDocs).catch((e) => { setDocs([]); setError(e.message) })
  useEffect(() => { load() }, [])

  async function upload(file) {
    if (!file) return
    setError('')
    setNotice('')
    setUploading(file.name)
    const form = new FormData()
    form.append('file', file)
    try {
      const doc = await request('/kb/upload', { method: 'POST', body: form })
      setNotice(`${doc.filename} is ready: split into ${doc.chunk_count} searchable passages.`)
    } catch (err) {
      setError(err.message) // the server explains unsupported types, empty files, size limits
    } finally {
      setUploading('')
      if (fileInput.current) fileInput.current.value = ''
      load()
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    upload(e.dataTransfer.files[0])
  }

  return (
    <section className="kb">
      <h1 className="display">Knowledge base</h1>
      <p className="lede">The agent answers only from these documents and cites them in the chat.</p>

      <label
        className={`dropzone ${dragging ? 'dragging' : ''} ${uploading ? 'busy' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={(e) => !e.currentTarget.contains(e.relatedTarget) && setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={fileInput}
          type="file"
          accept={ACCEPTED.join(',')}
          className="visually-hidden"
          onChange={(e) => upload(e.target.files[0])}
          disabled={!!uploading}
        />
        {uploading ? (
          <>
            <strong>Reading {uploading}…</strong>
            <span className="muted">Extracting text, splitting it into passages and indexing them.</span>
            <span className="progress" aria-hidden="true" />
          </>
        ) : (
          <>
            <strong>{dragging ? 'Drop to upload' : 'Drop a document here, or click to choose one'}</strong>
            <span className="muted">PDF, TXT or DOCX, up to 10 MB</span>
          </>
        )}
      </label>
      {error && <p className="error" role="alert">{error}</p>}
      {notice && <p className="success" role="status">{notice}</p>}

      <h2 className="section-title">Documents{docs?.length ? ` (${docs.length})` : ''}</h2>
      {docs === null && <p className="muted">Loading…</p>}
      {docs?.length === 0 && <p className="empty">Nothing here yet. Upload your FAQ or policy documents to get started.</p>}
      <ul className="doc-list">
        {docs?.map((d) => (
          <li key={d.id} className="doc">
            <span className="doc-ext">{d.filename.split('.').pop().toLowerCase()}</span>
            <div className="doc-main">
              <span className="doc-name">{d.filename}</span>
              <span className="muted small">
                Uploaded {formatDate(d.uploaded_at)}{d.status === 'Processed' && `, ${d.chunk_count} passages`}
              </span>
              {d.error && <span className="error small">{d.error}</span>}
            </div>
            <span className={`pill ${STATUS_PILL[d.status]}`}>{d.status}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}
