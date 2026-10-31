// Dev: the Vite server (:5173) calls uvicorn on :8000. A production build is served by the
// backend itself (see the Dockerfile), so it calls the server it was loaded from.
const API_URL = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? 'http://localhost:8000' : '')

// fetch that throws an Error with a readable message on network failure or non-2xx status.
async function send(path, options) {
  let res
  try {
    res = await fetch(API_URL + path, options)
  } catch {
    throw new Error(`Cannot reach the server at ${API_URL || location.origin}. Is the backend running?`)
  }
  if (!res.ok) {
    const detail = (await res.json().catch(() => null))?.detail
    // FastAPI validation errors (422) come back as a list of {loc, msg}.
    const message = Array.isArray(detail)
      ? detail.map((d) => `${d.loc?.at(-1)}: ${d.msg}`).join('; ')
      : detail || `Request failed (${res.status})`
    throw Object.assign(new Error(message), { status: res.status })
  }
  return res
}

const jsonOptions = (data, method = 'POST') =>
  ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) })

export const request = async (path, options = {}) => (await send(path, options)).json()

export const postJSON = (path, data, method = 'POST') => request(path, jsonOptions(data, method))

// POST, then read the JSON Lines response as it arrives, calling onEvent for each line.
export async function streamJSONLines(path, data, onEvent) {
  const res = await send(path, jsonOptions(data))
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value
    const lines = buffer.split('\n')
    buffer = lines.pop() // last piece may be an incomplete line
    lines.filter(Boolean).forEach((line) => onEvent(JSON.parse(line)))
  }
}

export const formatDate = (iso) =>
  new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
