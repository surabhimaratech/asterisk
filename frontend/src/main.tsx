import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

type Kind = 'book' | 'movie' | 'podcast' | 'article' | 'concept' | 'question' | 'thought'
type Item = { id: number; kind: Kind; title: string; note: string | null; url: string | null; created_at: string }
const kinds: Kind[] = ['book', 'movie', 'podcast', 'article', 'concept', 'question', 'thought']
const icons: Record<Kind,string> = { book:'📚', movie:'🎬', podcast:'🎧', article:'📰', concept:'💡', question:'❓', thought:'✨' }
const labels: Record<Kind,string> = { book:'Book', movie:'Movie', podcast:'Podcast', article:'Article', concept:'Concept', question:'Question', thought:'Random thought' }
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
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(true)
  async function refresh() {
    try {
      const r = await fetch('/api/items')
      if (!r.ok) throw new Error('Could not load library')
      setItems(await r.json()); setError('')
    } catch(e) { setError(e instanceof Error ? e.message : 'Something went wrong') }
    finally { setLoading(false) }
  }
  useEffect(() => { void refresh() }, [])
  async function save(e: React.FormEvent) {
    e.preventDefault()
    if (!form.title.trim()) return
    setBusy(true); setError('')
    try {
      const r = await fetch(editing === null ? '/api/items' : `/api/items/${editing}`, {
        method: editing === null ? 'POST' : 'PATCH', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind: form.kind, title: form.title.trim(), note: form.note || null, url: form.url || null })
      })
      if (!r.ok) throw new Error(await responseError(r, 'Could not save item'))
      const message = editing === null ? 'Curiosity saved.' : 'Changes saved.'
      setForm(empty); setEditing(null); await refresh()
      setSuccess(message)
      window.setTimeout(() => setSuccess(''), 2500)
    } catch(e) { setError(e instanceof Error ? e.message : 'Save failed') }
    finally { setBusy(false) }
  }
  function edit(item: Item) {
    setEditing(item.id)
    setForm({ kind: item.kind, title: item.title, note: item.note || '', url: item.url || '' })
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }
  async function remove(item: Item) {
    if (!window.confirm(`Delete “${item.title}”?`)) return
    try {
      const r = await fetch(`/api/items/${item.id}`, { method: 'DELETE' })
      if (!r.ok) throw new Error(await responseError(r, 'Could not delete item'))
      if (editing === item.id) { setEditing(null); setForm(empty) }
      await refresh()
      setSuccess('Curiosity deleted.')
      window.setTimeout(() => setSuccess(''), 2500)
    } catch(e) { setError(e instanceof Error ? e.message : 'Delete failed') }
  }
  const visible = items.filter(i => (filter === 'all' || i.kind === filter) && `${i.title} ${i.note || ''}`.toLowerCase().includes(query.toLowerCase()))
  return <main>
    <header><div className="eyebrow">A PERSONAL DISCOVERY LABORATORY</div><h1>Asterisk<span>✦</span></h1><p>There's more to the story.</p></header>
    <section className="panel capture"><div className="section-head"><h2>{editing === null ? 'Quick capture' : 'Edit curiosity'}</h2><span>01 / COLLECT</span></div>
      <form onSubmit={save}>
        <div className="kind-row" aria-label="Curiosity type">{kinds.map(k => <button key={k} type="button" aria-pressed={form.kind===k} className={'kind '+(form.kind===k?'selected':'')} onClick={()=>setForm({...form,kind:k})}>{icons[k]} {labels[k]}</button>)}</div>
        <label htmlFor="title">What caught your attention?</label><input id="title" autoFocus maxLength={240} placeholder="A book, an idea, a fascinating question…" value={form.title} onChange={e=>setForm({...form,title:e.target.value})} required />
        <label htmlFor="note">What intrigued you? <small>Optional</small></label><textarea id="note" rows={3} placeholder="A fleeting thought is enough." value={form.note} onChange={e=>setForm({...form,note:e.target.value})} />
        <label htmlFor="url">Link <small>Optional</small></label><input id="url" type="url" placeholder="https://…" value={form.url} onChange={e=>setForm({...form,url:e.target.value})} />
        <div className="actions"><button className="primary" disabled={busy || !form.title.trim()}>{busy?'Saving…':editing===null?'＋ Save curiosity':'Save changes'}</button>{editing!==null&&<button type="button" className="secondary" onClick={()=>{setEditing(null);setForm(empty)}}>Cancel</button>}</div>
      </form>
    </section>
    <section className="library"><div className="section-head"><div><h2>Curiosity library <em>{items.length}</em></h2><p>Everything that made you pause and wonder.</p></div><span>02 / EXPLORE</span></div>
      <div className="toolbar"><input aria-label="Search your library" placeholder="⌕  Search titles and notes" value={query} onChange={e=>setQuery(e.target.value)}/><select aria-label="Filter by type" value={filter} onChange={e=>setFilter(e.target.value as Kind|'all')}><option value="all">All types</option>{kinds.map(k=><option key={k} value={k}>{labels[k]}</option>)}</select></div>
      {loading?<div className="empty">Loading your library…</div>:visible.length===0?<div className="empty">{items.length===0?'Your library is waiting for its first curiosity. Save a book, question, or passing thought above.':'No curiosities match your search and filter.'}</div>:<div className="cards">{visible.map(i=><article className="item" key={i.id}><div className="item-top"><span className="pill">{icons[i.kind]} {labels[i.kind]}</span><span className="date">{new Date(i.created_at).toLocaleDateString()}</span></div><h3>{i.title}</h3>{i.note&&<p>{i.note}</p>}{i.url&&<a href={i.url} target="_blank" rel="noopener noreferrer">Open source ↗</a>}<div className="item-actions"><button onClick={()=>edit(i)}>Edit</button><button onClick={()=>void remove(i)}>Delete</button></div></article>)}</div>}
    </section>
    <section className="future"><div>✧ &nbsp;COMING NEXT</div><h2>Sidequests</h2><p>Follow unexpected connections and discover where your curiosity leads.</p><button disabled>✦ Sidequests — coming soon</button></section>
    {error&&<div className="error" role="alert">{error}</div>}
    {success&&<div className="success" role="status">{success}</div>}
    <footer>Asterisk · v0.1 · Your curiosity belongs to you.</footer>
  </main>
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>)
