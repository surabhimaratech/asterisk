import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

type Kind = 'book' | 'movie' | 'podcast' | 'article' | 'concept' | 'question' | 'thought'
type ViewMode = 'library' | 'index'
type Item = { id: number; kind: Kind; title: string; note: string | null; url: string | null; created_at: string }

const kinds: Kind[] = ['book', 'movie', 'podcast', 'article', 'concept', 'question', 'thought']
const icons: Record<Kind, string> = { book: '📚', movie: '🎬', podcast: '🎧', article: '📰', concept: '💡', question: '❓', thought: '✨' }
const labels: Record<Kind, string> = { book: 'Book', movie: 'Movie', podcast: 'Podcast', article: 'Article', concept: 'Concept', question: 'Question', thought: 'Random thought' }
const spineColors = ['#9f493e', '#426359', '#c48245', '#5d617d', '#846244', '#66754c', '#764c59', '#3f6475']
const empty = { kind: 'book' as Kind, title: '', note: '', url: '' }

async function responseError(response: Response, fallback: string) {
  try {
    const body = await response.json()
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail) && body.detail[0]?.msg) return body.detail[0].msg
  } catch { /* The fallback is clearer than an invalid error response. */ }
  return fallback
}

function App() {
  const [items, setItems] = useState<Item[]>([])
  const [form, setForm] = useState(empty)
  const [editing, setEditing] = useState<number | null>(null)
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<Kind | 'all'>('all')
  const [view, setView] = useState<ViewMode>(() => localStorage.getItem('asterisk-view') === 'index' ? 'index' : 'library')
  const [selectedBook, setSelectedBook] = useState<number | null>(null)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => { localStorage.setItem('asterisk-view', view) }, [view])

  async function refresh() {
    try {
      const response = await fetch('/api/items')
      if (!response.ok) throw new Error('Could not load library')
      setItems(await response.json())
      setError('')
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void refresh() }, [])

  async function save(event: React.FormEvent) {
    event.preventDefault()
    if (!form.title.trim()) return
    setBusy(true)
    setError('')
    try {
      const response = await fetch(editing === null ? '/api/items' : `/api/items/${editing}`, {
        method: editing === null ? 'POST' : 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind: form.kind, title: form.title.trim(), note: form.note || null, url: form.url || null }),
      })
      if (!response.ok) throw new Error(await responseError(response, 'Could not save item'))
      const message = editing === null ? 'Curiosity saved.' : 'Changes saved.'
      setForm(empty)
      setEditing(null)
      await refresh()
      setSuccess(message)
      window.setTimeout(() => setSuccess(''), 2500)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Save failed')
    } finally {
      setBusy(false)
    }
  }

  function edit(item: Item) {
    setEditing(item.id)
    setForm({ kind: item.kind, title: item.title, note: item.note || '', url: item.url || '' })
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  async function remove(item: Item) {
    if (!window.confirm(`Delete “${item.title}”?`)) return
    try {
      const response = await fetch(`/api/items/${item.id}`, { method: 'DELETE' })
      if (!response.ok) throw new Error(await responseError(response, 'Could not delete item'))
      if (editing === item.id) { setEditing(null); setForm(empty) }
      if (selectedBook === item.id) setSelectedBook(null)
      await refresh()
      setSuccess('Curiosity deleted.')
      window.setTimeout(() => setSuccess(''), 2500)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Delete failed')
    }
  }

  function startBookCapture() {
    setEditing(null)
    setForm({ ...empty, kind: 'book' })
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const normalizedQuery = query.trim().toLowerCase()
  const visible = items.filter(item =>
    (filter === 'all' || item.kind === filter) &&
    `${item.title} ${item.note || ''}`.toLowerCase().includes(normalizedQuery),
  )
  const books = visible.filter(item => item.kind === 'book')
  const selected = books.find(book => book.id === selectedBook) || null
  const deferredCount = visible.length - books.length

  return <main>
    <header>
      <div className="eyebrow">A PERSONAL DISCOVERY LABORATORY</div>
      <h1>Asterisk<span>✦</span></h1>
      <p>There's more to the story.</p>
    </header>

    <section className="panel capture">
      <div className="section-head"><h2>{editing === null ? 'Quick capture' : 'Edit curiosity'}</h2><span>01 / COLLECT</span></div>
      <form onSubmit={save}>
        <div className="kind-row" aria-label="Curiosity type">
          {kinds.map(kind => <button key={kind} type="button" aria-pressed={form.kind === kind} className={`kind ${form.kind === kind ? 'selected' : ''}`} onClick={() => setForm({ ...form, kind })}>{icons[kind]} {labels[kind]}</button>)}
        </div>
        <label htmlFor="title">What caught your attention?</label>
        <input id="title" autoFocus maxLength={240} placeholder="A book, an idea, a fascinating question…" value={form.title} onChange={event => setForm({ ...form, title: event.target.value })} required />
        <label htmlFor="note">What intrigued you? <small>Optional</small></label>
        <textarea id="note" rows={3} placeholder="A fleeting thought is enough." value={form.note} onChange={event => setForm({ ...form, note: event.target.value })} />
        <label htmlFor="url">Link <small>Optional</small></label>
        <input id="url" type="url" placeholder="https://…" value={form.url} onChange={event => setForm({ ...form, url: event.target.value })} />
        <div className="actions">
          <button className="primary" disabled={busy || !form.title.trim()}>{busy ? 'Saving…' : editing === null ? '＋ Save curiosity' : 'Save changes'}</button>
          {editing !== null && <button type="button" className="secondary" onClick={() => { setEditing(null); setForm(empty) }}>Cancel</button>}
        </div>
      </form>
    </section>

    <section className="library">
      <div className="section-head"><div><h2>Curiosity library <em>{items.length}</em></h2><p>Everything that made you pause and wonder.</p></div><span>02 / EXPLORE</span></div>
      <div className="library-tools">
        <div className="search-wrap">
          <span aria-hidden="true">⌕</span>
          <input aria-label="Search your library" placeholder="Search titles and notes" value={query} onChange={event => setQuery(event.target.value)} />
          {query && <button type="button" aria-label="Clear search" onClick={() => setQuery('')}>×</button>}
        </div>
        <div className="view-toggle" aria-label="Library display" role="group">
          <button type="button" className={view === 'library' ? 'active' : ''} aria-pressed={view === 'library'} onClick={() => setView('library')}>▥ Library</button>
          <button type="button" className={view === 'index' ? 'active' : ''} aria-pressed={view === 'index'} onClick={() => setView('index')}>☷ Index</button>
        </div>
      </div>
      <div className="filter-row" aria-label="Filter library">
        <button type="button" className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>All <span>{items.length}</span></button>
        {kinds.map(kind => <button type="button" key={kind} className={filter === kind ? 'active' : ''} onClick={() => setFilter(kind)}>{labels[kind]} <span>{items.filter(item => item.kind === kind).length}</span></button>)}
      </div>

      {loading ? <div className="empty">Loading your library…</div> : visible.length === 0 ?
        <div className="empty">{items.length === 0 ? 'Your library is waiting for its first curiosity. Save a book, question, or passing thought above.' : 'No curiosities match your search and filter.'}</div> :
        view === 'index' ? <div className="cards">{visible.map(item => <ItemCard key={item.id} item={item} onEdit={edit} onRemove={remove} />)}</div> :
        <div className="book-library">
          {books.length > 0 ? <>
            <div className="shelf-heading"><div><span>THE BOOKSHELF</span><h3>{normalizedQuery ? `${books.length} matching ${books.length === 1 ? 'book' : 'books'}` : `${books.length} ${books.length === 1 ? 'volume' : 'volumes'}`}</h3></div><p>Select a spine to take a closer look.</p></div>
            <div className="shelf" aria-label="Bookshelf">
              <div className="shelf-books">
                {books.map((book, index) => <button
                  type="button"
                  key={book.id}
                  className={`book-spine ${selectedBook === book.id ? 'selected' : ''}`}
                  style={{ '--spine-color': spineColors[book.id % spineColors.length], '--book-height': `${172 + ((book.id * 17 + index * 11) % 58)}px`, '--book-width': `${45 + ((book.title.length + book.id) % 24)}px` } as React.CSSProperties}
                  aria-label={`${book.title}, added ${new Date(book.created_at).toLocaleDateString()}`}
                  aria-pressed={selectedBook === book.id}
                  onClick={() => setSelectedBook(selectedBook === book.id ? null : book.id)}
                ><span className="spine-star">✦</span><span className="spine-title">{book.title}</span></button>)}
              </div>
              <div className="shelf-edge" />
            </div>
            {selected && <BookDetail item={selected} onClose={() => setSelectedBook(null)} onEdit={edit} onRemove={remove} />}
          </> : <div className="empty bookshelf-empty"><span>📖</span><strong>No books on this shelf yet.</strong><button type="button" onClick={startBookCapture}>Add your first book</button></div>}
          {deferredCount > 0 && <div className="index-nudge"><span>✦</span><p><strong>{deferredCount} other {deferredCount === 1 ? 'curiosity is' : 'curiosities are'} nearby.</strong> Their visual collections are still taking shape.</p><button type="button" onClick={() => setView('index')}>View in Index</button></div>}
        </div>}
    </section>

    <section className="future"><div>✧ &nbsp;COMING NEXT</div><h2>Sidequests</h2><p>Follow unexpected connections and discover where your curiosity leads.</p><button disabled>✦ Sidequests — coming soon</button></section>
    {error && <div className="error" role="alert">{error}</div>}
    {success && <div className="success" role="status">{success}</div>}
    <footer>Asterisk · v0.1 · Your curiosity belongs to you.</footer>
  </main>
}

function ItemCard({ item, onEdit, onRemove }: { item: Item; onEdit: (item: Item) => void; onRemove: (item: Item) => void }) {
  return <article className="item">
    <div className="item-top"><span className="pill">{icons[item.kind]} {labels[item.kind]}</span><span className="date">{new Date(item.created_at).toLocaleDateString()}</span></div>
    <h3>{item.title}</h3>
    {item.note && <p>{item.note}</p>}
    {item.url && <a href={item.url} target="_blank" rel="noopener noreferrer">Open source ↗</a>}
    <div className="item-actions"><button onClick={() => onEdit(item)}>Edit</button><button onClick={() => void onRemove(item)}>Delete</button></div>
  </article>
}

function BookDetail({ item, onClose, onEdit, onRemove }: { item: Item; onClose: () => void; onEdit: (item: Item) => void; onRemove: (item: Item) => void }) {
  return <article className="book-detail">
    <button type="button" className="detail-close" aria-label="Close book details" onClick={onClose}>×</button>
    <div className="bookplate">FROM THE LIBRARY OF<br /><strong>ASTERISK</strong></div>
    <div className="book-detail-copy">
      <span>BOOK · ADDED {new Date(item.created_at).toLocaleDateString()}</span>
      <h3>{item.title}</h3>
      {item.note ? <p>{item.note}</p> : <p className="muted">No personal note yet.</p>}
      {item.url && <a href={item.url} target="_blank" rel="noopener noreferrer">Open source ↗</a>}
      <div className="item-actions"><button onClick={() => onEdit(item)}>Edit this book</button><button onClick={() => void onRemove(item)}>Remove from shelf</button></div>
    </div>
  </article>
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>)
