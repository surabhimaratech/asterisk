import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

type Kind = 'book' | 'movie' | 'podcast' | 'article' | 'concept' | 'question' | 'thought'
type ViewMode = 'library' | 'index'
type Item = { id: number; kind: Kind; title: string; note: string | null; url: string | null; created_at: string }
type ItemConcept = { id: number; item_id: number; canonical_concept_id: number | null; name: string; category: 'personal' | 'general' | 'mechanism'; origin: 'ai' | 'user'; explanation: string | null; level: 'topic' | 'concept' | 'mechanism' | null; relationship: string | null; evidence_status: 'supported' | 'inferred' | 'unverified' | null; confidence: number | null; source_ids: number[]; canonical_name: string | null; definition: string | null; domain: string | null; mechanism: string | null; relationship_type: 'explicit' | 'illustrated' | 'interpretive' | null; relationship_evidence: string | null; concept_verification: string | null; relationship_verification: string | null; created_at: string }
type ResearchSource = { id: number; url: string; title: string; publisher: string | null; source_type: string; excerpt: string; retrieval_status: string; retrieved_at: string }
type Enrichment = { status: 'not_researched' | 'researched' | 'needs_confirmation'; resolved_title: string | null; creator: string | null; release_info: string | null; identity_summary: string | null; confirmation_question: string | null; researched_at: string | null; sources: ResearchSource[]; concepts: ItemConcept[]; rejected_leads: { name: string; accepted: boolean; rejection_reason: string }[] }
type Sidequest = { id: number; item_a_id: number; item_b_id: number; item_a_title: string; item_b_title: string; title: string; connection: string; bridge: string; why_interesting: string; where_it_breaks: string; new_question: string; evidence_status: string; concepts_used: string[]; evidence: string[]; created_at: string }

const kinds: Kind[] = ['book', 'movie', 'podcast', 'article', 'concept', 'question', 'thought']
const icons: Record<Kind, string> = { book: '📚', movie: '🎬', podcast: '🎧', article: '📰', concept: '💡', question: '❓', thought: '✨' }
const labels: Record<Kind, string> = { book: 'Book', movie: 'Movie', podcast: 'Podcast', article: 'Article', concept: 'Concept', question: 'Question', thought: 'Random thought' }
const spineColors = ['#9f493e', '#426359', '#c48245', '#5d617d', '#846244', '#66754c', '#764c59', '#3f6475']
const coverPalettes = [
  ['#e96b4a', '#301e32'], ['#e7bd57', '#253e48'], ['#78a79a', '#202e3b'], ['#c47d96', '#3d2441'],
  ['#d6d0b8', '#963f32'], ['#7788aa', '#241e30'], ['#d99959', '#394a38'], ['#a4c5cc', '#733d4b'],
]
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
  const [focusedZone, setFocusedZone] = useState<Kind | null>(null)
  const [selectedBook, setSelectedBook] = useState<number | null>(null)
  const [selectedMovie, setSelectedMovie] = useState<number | null>(null)
  const [selectedPodcast, setSelectedPodcast] = useState<number | null>(null)
  const [selectedArticle, setSelectedArticle] = useState<number | null>(null)
  const [selectedConcept, setSelectedConcept] = useState<number | null>(null)
  const [selectedQuestion, setSelectedQuestion] = useState<number | null>(null)
  const [selectedThought, setSelectedThought] = useState<number | null>(null)
  const [conceptItem, setConceptItem] = useState<Item | null>(null)
  const [conceptCounts, setConceptCounts] = useState<Record<number, number>>({})
  const [sidequests, setSidequests] = useState<Sidequest[]>([])
  const [sidequestBusy, setSidequestBusy] = useState(false)
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

  async function refreshSidequests() {
    try {
      const response = await fetch('/api/sidequests')
      if (response.ok) setSidequests(await response.json())
    } catch { /* The main library remains usable if Sidequests cannot load. */ }
  }

  useEffect(() => { void refreshSidequests() }, [])

  useEffect(() => {
    if (items.length === 0) { setConceptCounts({}); return }
    let cancelled = false
    void Promise.all(items.map(async item => {
      const response = await fetch(`/api/items/${item.id}/concepts`)
      return [item.id, response.ok ? ((await response.json()) as ItemConcept[]).length : 0] as const
    })).then(entries => { if (!cancelled) setConceptCounts(Object.fromEntries(entries)) })
      .catch(() => { /* Readiness is helpful context, not required for using the library. */ })
    return () => { cancelled = true }
  }, [items])

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
      if (selectedMovie === item.id) setSelectedMovie(null)
      if (selectedPodcast === item.id) setSelectedPodcast(null)
      if (selectedArticle === item.id) setSelectedArticle(null)
      if (selectedConcept === item.id) setSelectedConcept(null)
      if (selectedQuestion === item.id) setSelectedQuestion(null)
      if (selectedThought === item.id) setSelectedThought(null)
      await refresh()
      setSuccess('Curiosity deleted.')
      window.setTimeout(() => setSuccess(''), 2500)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Delete failed')
    }
  }

  function startCapture(kind: Kind) {
    setEditing(null)
    setForm({ ...empty, kind })
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  function focusZone(kind: Kind) {
    setFocusedZone(kind)
    setFilter(kind)
    window.setTimeout(() => document.getElementById('focused-collection')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50)
  }

  function returnToRoom() {
    setFocusedZone(null)
    setFilter('all')
    window.setTimeout(() => document.getElementById('curiosity-room')?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 50)
  }

  function exploreConcepts(item: Item) {
    setConceptItem(item)
    window.setTimeout(() => document.getElementById('concept-workbench')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50)
  }

  async function surpriseMe() {
    setSidequestBusy(true)
    setError('')
    try {
      const response = await fetch('/api/sidequests/surprise', { method: 'POST' })
      if (!response.ok) throw new Error(await responseError(response, 'Could not create a Sidequest'))
      const created = await response.json()
      setSidequests(current => [created, ...current])
      setSuccess('A new Sidequest sparked.')
      window.setTimeout(() => setSuccess(''), 2500)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Sidequest failed')
    } finally {
      setSidequestBusy(false)
    }
  }

  const normalizedQuery = query.trim().toLowerCase()
  const visible = items.filter(item =>
    (filter === 'all' || item.kind === filter) &&
    `${item.title} ${item.note || ''}`.toLowerCase().includes(normalizedQuery),
  )
  const books = visible.filter(item => item.kind === 'book')
  const movies = visible.filter(item => item.kind === 'movie')
  const podcasts = visible.filter(item => item.kind === 'podcast')
  const articles = visible.filter(item => item.kind === 'article')
  const concepts = visible.filter(item => item.kind === 'concept')
  const questions = visible.filter(item => item.kind === 'question')
  const thoughts = visible.filter(item => item.kind === 'thought')
  const selectedBookItem = books.find(book => book.id === selectedBook) || null
  const selectedMovieItem = movies.find(movie => movie.id === selectedMovie) || null
  const selectedPodcastItem = podcasts.find(podcast => podcast.id === selectedPodcast) || null
  const selectedArticleItem = articles.find(article => article.id === selectedArticle) || null
  const selectedConceptItem = concepts.find(concept => concept.id === selectedConcept) || null
  const selectedQuestionItem = questions.find(question => question.id === selectedQuestion) || null
  const selectedThoughtItem = thoughts.find(thought => thought.id === selectedThought) || null
  const selectedItem = selectedBookItem || selectedMovieItem || selectedPodcastItem || selectedArticleItem || selectedConceptItem || selectedQuestionItem || selectedThoughtItem
  const deferredCount = visible.length - books.length - movies.length - podcasts.length - articles.length - concepts.length - questions.length - thoughts.length

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
        <button type="button" className={filter === 'all' ? 'active' : ''} onClick={returnToRoom}>All <span>{items.length}</span></button>
        {kinds.map(kind => <button type="button" key={kind} className={filter === kind ? 'active' : ''} onClick={() => focusZone(kind)}>{labels[kind]} <span>{items.filter(item => item.kind === kind).length}</span></button>)}
      </div>

      {loading ? <div className="empty">Loading your library…</div> : view === 'index' ? (visible.length === 0 ?
        <div className="empty">{items.length === 0 ? 'Your library is waiting for its first curiosity. Save a book, question, or passing thought above.' : 'No curiosities match your search and filter.'}</div> :
        <div className="cards">{visible.map(item => <ItemCard key={item.id} item={item} onEdit={edit} onRemove={remove} onExplore={exploreConcepts} />)}</div>) : <>
        <RoomShell items={items} query={query} focusedZone={focusedZone} onFocus={focusZone} />
        {focusedZone && <div className="room-focus-nav"><button type="button" onClick={returnToRoom}>← Back to room</button><span>{labels[focusedZone]} collection</span></div>}
        {!focusedZone && <div className="room-invitation"><span>✦</span><p>Wander in. Choose a corner of the room to look closer.</p></div>}
        {focusedZone && <div className="collection-library" id="focused-collection">
          {(filter === 'all' || filter === 'book') && (books.length > 0 ? <section className="visual-collection book-collection">
            <div className="shelf-heading"><div><span>THE BOOKSHELF</span><h3>{normalizedQuery ? `${books.length} matching ${books.length === 1 ? 'book' : 'books'}` : `${books.length} ${books.length === 1 ? 'volume' : 'volumes'}`}</h3></div><p>Select a spine to take a closer look.</p></div>
            <div className="shelf" aria-label="Bookshelf"><div className="shelf-books">
              {books.map((book, index) => <button type="button" key={book.id} className={`book-spine ${selectedBook === book.id ? 'selected' : ''}`} style={{ '--spine-color': spineColors[book.id % spineColors.length], '--book-height': `${172 + ((book.id * 17 + index * 11) % 58)}px`, '--book-width': `${45 + ((book.title.length + book.id) % 24)}px` } as React.CSSProperties} aria-label={`${book.title}, added ${new Date(book.created_at).toLocaleDateString()}`} aria-pressed={selectedBook === book.id} onClick={() => setSelectedBook(selectedBook === book.id ? null : book.id)}><span className="spine-star">✦</span><span className="spine-title">{book.title}</span></button>)}
            </div><div className="shelf-edge" /></div>
            {selectedBookItem && <BookDetail item={selectedBookItem} onClose={() => setSelectedBook(null)} onEdit={edit} onRemove={remove} />}
          </section> : filter === 'book' && <CollectionEmpty icon="📖" message="No books on this shelf yet." action="Add your first book" onAction={() => startCapture('book')} />)}

          {(filter === 'all' || filter === 'movie') && (movies.length > 0 ? <section className="visual-collection movie-collection">
            <div className="shelf-heading media-heading"><div><span>THE MEDIA SHELF</span><h3>{normalizedQuery ? `${movies.length} matching ${movies.length === 1 ? 'movie' : 'movies'}` : `${movies.length} ${movies.length === 1 ? 'title' : 'titles'}`}</h3></div><p>Choose a case for tonight's feature.</p></div>
            <div className="media-shelf" aria-label="Movie shelf"><div className="movie-cases">
              {movies.map(movie => <MovieCase key={movie.id} item={movie} selected={selectedMovie === movie.id} onSelect={() => setSelectedMovie(selectedMovie === movie.id ? null : movie.id)} />)}
            </div><div className="media-shelf-edge" /></div>
            {selectedMovieItem && <MovieDetail item={selectedMovieItem} onClose={() => setSelectedMovie(null)} onEdit={edit} onRemove={remove} />}
          </section> : filter === 'movie' && <CollectionEmpty icon="🎞️" message="No movies on this shelf yet." action="Add your first movie" onAction={() => startCapture('movie')} />)}

          {(filter === 'all' || filter === 'podcast') && (podcasts.length > 0 ? <section className="visual-collection podcast-collection">
            <div className="shelf-heading radio-heading"><div><span>ON THE AIR</span><h3>{normalizedQuery ? `${podcasts.length} matching ${podcasts.length === 1 ? 'show' : 'shows'}` : `${podcasts.length} ${podcasts.length === 1 ? 'recording' : 'recordings'}`}</h3></div><p>Press play on something worth hearing.</p></div>
            <div className="cassette-tray" aria-label="Podcast cassette collection">
              {podcasts.map(podcast => <PodcastCassette key={podcast.id} item={podcast} selected={selectedPodcast === podcast.id} onSelect={() => setSelectedPodcast(selectedPodcast === podcast.id ? null : podcast.id)} />)}
            </div>
            {selectedPodcastItem && <PodcastDetail item={selectedPodcastItem} onClose={() => setSelectedPodcast(null)} onEdit={edit} onRemove={remove} />}
          </section> : filter === 'podcast' && <CollectionEmpty icon="📻" message="No recordings in the tray yet." action="Add your first podcast" onAction={() => startCapture('podcast')} />)}

          {(filter === 'all' || filter === 'article') && (articles.length > 0 ? <section className="visual-collection article-collection">
            <div className="shelf-heading press-heading"><div><span>THE READING PILE</span><h3>{normalizedQuery ? `${articles.length} matching ${articles.length === 1 ? 'article' : 'articles'}` : `${articles.length} ${articles.length === 1 ? 'clipping' : 'clippings'}`}</h3></div><p>Saved for a quiet moment.</p></div>
            <div className="article-pile" aria-label="Article reading pile">
              {articles.map(article => <ArticleClipping key={article.id} item={article} selected={selectedArticle === article.id} onSelect={() => setSelectedArticle(selectedArticle === article.id ? null : article.id)} />)}
            </div>
            {selectedArticleItem && <ArticleDetail item={selectedArticleItem} onClose={() => setSelectedArticle(null)} onEdit={edit} onRemove={remove} />}
          </section> : filter === 'article' && <CollectionEmpty icon="🗞️" message="The reading pile is empty." action="Save your first article" onAction={() => startCapture('article')} />)}

          {(filter === 'all' || filter === 'concept') && (concepts.length > 0 ? <section className="visual-collection concept-collection">
            <div className="shelf-heading catalog-heading"><div><span>THE CARD CATALOG</span><h3>{normalizedQuery ? `${concepts.length} matching ${concepts.length === 1 ? 'concept' : 'concepts'}` : `${concepts.length} ${concepts.length === 1 ? 'entry' : 'entries'}`}</h3></div><p>Ideas filed for future reference.</p></div>
            <div className="catalog-cabinet" aria-label="Concept card catalog"><div className="catalog-drawer"><div className="concept-cards">
              {concepts.map(concept => <ConceptCard key={concept.id} item={concept} selected={selectedConcept === concept.id} onSelect={() => setSelectedConcept(selectedConcept === concept.id ? null : concept.id)} />)}
            </div><div className="drawer-front"><span>CONCEPTS</span><i /></div></div></div>
            {selectedConceptItem && <ConceptDetail item={selectedConceptItem} onClose={() => setSelectedConcept(null)} onEdit={edit} onRemove={remove} />}
          </section> : filter === 'concept' && <CollectionEmpty icon="🗂️" message="No concepts have been filed yet." action="File your first concept" onAction={() => startCapture('concept')} />)}

          {(filter === 'all' || filter === 'question') && (questions.length > 0 ? <section className="visual-collection question-collection">
            <div className="shelf-heading question-heading"><div><span>THE QUESTION BOX</span><h3>{normalizedQuery ? `${questions.length} matching ${questions.length === 1 ? 'question' : 'questions'}` : `${questions.length} ${questions.length === 1 ? 'bright idea' : 'bright ideas'}`}</h3></div><p>Unanswered is an excellent place to begin.</p></div>
            <div className="question-box" aria-label="Question lightbulb box"><div className="bulb-compartments">
              {questions.map(question => <QuestionBulb key={question.id} item={question} selected={selectedQuestion === question.id} onSelect={() => setSelectedQuestion(selectedQuestion === question.id ? null : question.id)} />)}
            </div><div className="box-label">FRAGILE · IDEAS INSIDE · ✦</div></div>
            {selectedQuestionItem && <QuestionDetail item={selectedQuestionItem} onClose={() => setSelectedQuestion(null)} onEdit={edit} onRemove={remove} />}
          </section> : filter === 'question' && <CollectionEmpty icon="💡" message="The question box is waiting." action="Ask your first question" onAction={() => startCapture('question')} />)}

          {(filter === 'all' || filter === 'thought') && (thoughts.length > 0 ? <section className="visual-collection thought-collection">
            <div className="shelf-heading thought-heading"><div><span>ODDS & ENDS</span><h3>{normalizedQuery ? `${thoughts.length} matching ${thoughts.length === 1 ? 'thought' : 'thoughts'}` : `${thoughts.length} ${thoughts.length === 1 ? 'scrap' : 'scraps'}`}</h3></div><p>Nothing is too small to keep.</p></div>
            <div className="thought-pocket" aria-label="Random thought scraps"><div className="thought-scraps">
              {thoughts.map(thought => <ThoughtScrap key={thought.id} item={thought} selected={selectedThought === thought.id} onSelect={() => setSelectedThought(selectedThought === thought.id ? null : thought.id)} />)}
            </div><div className="pocket-front"><span>LOOSE THOUGHTS</span><small>✦ KEEP FOR LATER</small></div></div>
            {selectedThoughtItem && <ThoughtDetail item={selectedThoughtItem} onClose={() => setSelectedThought(null)} onEdit={edit} onRemove={remove} />}
          </section> : filter === 'thought' && <CollectionEmpty icon="📝" message="No loose thoughts tucked away yet." action="Capture a passing thought" onAction={() => startCapture('thought')} />)}

          {deferredCount > 0 && <div className="index-nudge"><span>✦</span><p><strong>{deferredCount} other {deferredCount === 1 ? 'curiosity is' : 'curiosities are'} nearby.</strong> Their visual collections are still taking shape.</p><button type="button" onClick={() => setView('index')}>View in Index</button></div>}
        </div>}</>}
    </section>

    {selectedItem && <div className="concept-callout"><span>✦ Look beneath the surface</span><button type="button" onClick={() => exploreConcepts(selectedItem)}>Explore Concepts</button></div>}
    {conceptItem && <ConceptExplorer item={conceptItem} onClose={() => setConceptItem(null)} onError={setError} onChange={count => setConceptCounts(current => ({ ...current, [conceptItem.id]: count }))} />}

    <SidequestsView items={items} conceptCounts={conceptCounts} sidequests={sidequests} busy={sidequestBusy} onExplore={exploreConcepts} onSurprise={() => void surpriseMe()} />
    {error && <div className="error" role="alert">{error}</div>}
    {success && <div className="success" role="status">{success}</div>}
    <footer>Asterisk · v0.2 · Your curiosity belongs to you.</footer>
  </main>
}

function ItemCard({ item, onEdit, onRemove, onExplore }: { item: Item; onEdit: (item: Item) => void; onRemove: (item: Item) => void; onExplore: (item: Item) => void }) {
  return <article className="item">
    <div className="item-top"><span className="pill">{icons[item.kind]} {labels[item.kind]}</span><span className="date">{new Date(item.created_at).toLocaleDateString()}</span></div>
    <h3>{item.title}</h3>
    {item.note && <p>{item.note}</p>}
    {item.url && <a href={item.url} target="_blank" rel="noopener noreferrer">Open source ↗</a>}
    <div className="item-actions"><button onClick={() => onExplore(item)}>Explore Concepts</button><button onClick={() => onEdit(item)}>Edit</button><button onClick={() => void onRemove(item)}>Delete</button></div>
  </article>
}

function RoomShell({ items, query, focusedZone, onFocus }: { items: Item[]; query: string; focusedZone: Kind | null; onFocus: (kind: Kind) => void }) {
  const normalizedRoomQuery = query.trim().toLowerCase()
  const matchesRoomQuery = (item: Item) => !normalizedRoomQuery || `${item.title} ${item.note || ''}`.toLowerCase().includes(normalizedRoomQuery)
  const miniClass = (item: Item) => `mini-object ${matchesRoomQuery(item) ? 'match' : 'nonmatch'}`
  const roomItems = (kind: Kind, limit: number) => items.filter(item => item.kind === kind).slice(0, limit)
  const resultCount = (kind: Kind) => items.filter(item => item.kind === kind && matchesRoomQuery(item)).length
  const zoneClass = (kind: Kind) => `room-zone zone-${kind} ${focusedZone === kind ? 'active' : ''} ${normalizedRoomQuery && resultCount(kind) === 0 ? 'no-match' : ''}`
  const roomBooks = roomItems('book', 18)
  const roomMovies = roomItems('movie', 5)
  const roomPodcasts = roomItems('podcast', 3)
  const roomArticles = roomItems('article', 4)
  const roomConcepts = roomItems('concept', 5)
  const roomQuestions = roomItems('question', 4)
  const roomThoughts = roomItems('thought', 5)
  return <section className={`room-shell ${normalizedRoomQuery ? 'searching' : ''}`} id="curiosity-room" aria-label="Asterisk library room viewed from the doorway">
    <div className="room-glow" />
    <div className="room-lights" aria-hidden="true"><i /><i /><i /><i /><i /></div>
    <div className="left-wall"><div className="media-console"><div className="console-screen"><div className="mini-movies">{roomMovies.map(movie => <i className={miniClass(movie)} key={movie.id} title={movie.title} style={{ background: coverPalettes[movie.id % coverPalettes.length][0] }} />)}</div></div><div className="console-doors"><i /><i /></div></div><div className="radio-stand"><div className="mini-tapes">{roomPodcasts.map(podcast => <i className={miniClass(podcast)} key={podcast.id} title={podcast.title} />)}</div><div className="room-radio"><i /><b /></div></div></div>
    <div className="back-wall"><div className="room-emblem">✦<small>ASTERISK</small></div><div className="room-bookcase"><div className="case-top" /><div className="case-bays"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div><div className="miniature-books">{roomBooks.map(book => <i className={miniClass(book)} key={book.id} title={book.title} style={{ '--mini-book': spineColors[book.id % spineColors.length], height: `${24 + (book.id % 13)}px` } as React.CSSProperties} />)}</div><div className="case-base" /></div><div className="room-catalog"><i /><i /><i /><i />{roomConcepts.map(concept => <b className={miniClass(concept)} key={concept.id} title={concept.title}>{concept.title.slice(0, 1)}</b>)}</div></div>
    <div className="right-wall"><div className="room-desk"><div className="mini-papers">{roomArticles.map(article => <i className={miniClass(article)} key={article.id} title={article.title} />)}</div><div className="mini-thoughts">{roomThoughts.map(thought => <i className={miniClass(thought)} key={thought.id} title={thought.title}>✦</i>)}</div><div className="desk-top" /><div className="desk-drawers"><i /><i /></div><div className="desk-legs"><i /><i /></div></div><div className="question-crate"><div className="mini-bulbs">{roomQuestions.map(question => <i className={miniClass(question)} key={question.id} title={question.title}>?</i>)}</div></div></div>
    <div className="room-plant plant-left" aria-hidden="true"><i /><i /><i /><i /><b /></div>
    <div className="room-plant plant-right" aria-hidden="true"><i /><i /><i /><i /><b /></div>
    <div className="floor-lamp" aria-hidden="true"><i /><b /></div>
    <div className="room-floor"><div className="room-rug"><span>THERE'S MORE<br />TO THE STORY</span></div></div>
    <div className="door-threshold"><span>THE CURIOSITY ROOM</span></div>
    {kinds.map(kind => <button type="button" key={kind} className={zoneClass(kind)} disabled={Boolean(normalizedRoomQuery) && resultCount(kind) === 0} onClick={() => onFocus(kind)} aria-label={`Open ${labels[kind]} collection, ${resultCount(kind)} ${normalizedRoomQuery ? 'matching' : 'saved'}`}><span>{labels[kind]}</span><b>{resultCount(kind)}</b></button>)}
  </section>
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

function MovieCase({ item, selected, onSelect }: { item: Item; selected: boolean; onSelect: () => void }) {
  const palette = coverPalettes[item.id % coverPalettes.length]
  const format = item.id % 3 === 0 ? 'VHS' : 'DVD'
  return <button
    type="button"
    className={`movie-case ${format === 'VHS' ? 'vhs' : ''} ${selected ? 'selected' : ''}`}
    style={{ '--cover-accent': palette[0], '--cover-ink': palette[1] } as React.CSSProperties}
    aria-label={`${item.title}, ${format} case, added ${new Date(item.created_at).toLocaleDateString()}`}
    aria-pressed={selected}
    onClick={onSelect}
  >
    <span className="case-format">{format}</span>
    <span className="cover-orbit" aria-hidden="true">✦</span>
    <span className="cover-title">{item.title}</span>
    <span className="cover-footer">AN ASTERISK SELECTION</span>
  </button>
}

function MovieDetail({ item, onClose, onEdit, onRemove }: { item: Item; onClose: () => void; onEdit: (item: Item) => void; onRemove: (item: Item) => void }) {
  return <article className="movie-detail">
    <button type="button" className="detail-close" aria-label="Close movie details" onClick={onClose}>×</button>
    <div className="ticket-stub"><span>✦ ASTERISK PICTURES</span><strong>ADMIT ONE</strong><small>{new Date(item.created_at).toLocaleDateString()}</small></div>
    <div className="movie-detail-copy">
      <span>MOVIE · FROM THE MEDIA SHELF</span>
      <h3>{item.title}</h3>
      {item.note ? <p>{item.note}</p> : <p className="muted">No personal note yet.</p>}
      {item.url && <a href={item.url} target="_blank" rel="noopener noreferrer">Open source ↗</a>}
      <div className="item-actions"><button onClick={() => onEdit(item)}>Edit this movie</button><button onClick={() => void onRemove(item)}>Remove from shelf</button></div>
    </div>
  </article>
}

function PodcastCassette({ item, selected, onSelect }: { item: Item; selected: boolean; onSelect: () => void }) {
  const palette = coverPalettes[(item.id + 3) % coverPalettes.length]
  return <button type="button" className={`cassette ${selected ? 'selected' : ''}`} style={{ '--tape-color': palette[0], '--tape-ink': palette[1] } as React.CSSProperties} aria-label={`${item.title}, podcast cassette, added ${new Date(item.created_at).toLocaleDateString()}`} aria-pressed={selected} onClick={onSelect}>
    <span className="cassette-label"><small>ASTERISK RADIO · EP {String(item.id).padStart(2, '0')}</small><strong>{item.title}</strong></span>
    <span className="tape-window" aria-hidden="true"><i /><b>60</b><i /></span>
    <span className="cassette-notches" aria-hidden="true">● &nbsp; ● &nbsp; ●</span>
  </button>
}

function PodcastDetail({ item, onClose, onEdit, onRemove }: { item: Item; onClose: () => void; onEdit: (item: Item) => void; onRemove: (item: Item) => void }) {
  return <article className="podcast-detail">
    <button type="button" className="detail-close" aria-label="Close podcast details" onClick={onClose}>×</button>
    <div className="radio-card"><span>ASTERISK<br />RADIO</span><strong>◉</strong><small>NOW PLAYING</small></div>
    <div className="podcast-detail-copy"><span>PODCAST · SAVED {new Date(item.created_at).toLocaleDateString()}</span><h3>{item.title}</h3>{item.note ? <p>{item.note}</p> : <p className="muted">No listening note yet.</p>}{item.url && <a href={item.url} target="_blank" rel="noopener noreferrer">Open source ↗</a>}<div className="item-actions"><button onClick={() => onEdit(item)}>Edit this podcast</button><button onClick={() => void onRemove(item)}>Remove recording</button></div></div>
  </article>
}

function ArticleClipping({ item, selected, onSelect }: { item: Item; selected: boolean; onSelect: () => void }) {
  return <button type="button" className={`article-clipping ${selected ? 'selected' : ''}`} aria-label={`${item.title}, article clipping, added ${new Date(item.created_at).toLocaleDateString()}`} aria-pressed={selected} onClick={onSelect}>
    <span className="paperclip" aria-hidden="true" />
    <span className="publication">THE ASTERISK READER <b>·</b> {new Date(item.created_at).toLocaleDateString()}</span>
    <strong>{item.title}</strong>
    <span className="article-rule" />
    <small>{item.note || 'Saved to read and remember.'}</small>
    <span className="folded-corner" aria-hidden="true" />
  </button>
}

function ArticleDetail({ item, onClose, onEdit, onRemove }: { item: Item; onClose: () => void; onEdit: (item: Item) => void; onRemove: (item: Item) => void }) {
  return <article className="article-detail">
    <button type="button" className="detail-close" aria-label="Close article details" onClick={onClose}>×</button>
    <div className="press-mark"><span>THE</span><strong>✦</strong><span>READER</span></div>
    <div className="article-detail-copy"><span>ARTICLE · CLIPPED {new Date(item.created_at).toLocaleDateString()}</span><h3>{item.title}</h3>{item.note ? <p>{item.note}</p> : <p className="muted">No margin note yet.</p>}{item.url && <a href={item.url} target="_blank" rel="noopener noreferrer">Read original ↗</a>}<div className="item-actions"><button onClick={() => onEdit(item)}>Edit this clipping</button><button onClick={() => void onRemove(item)}>Discard clipping</button></div></div>
  </article>
}

function ConceptCard({ item, selected, onSelect }: { item: Item; selected: boolean; onSelect: () => void }) {
  return <button type="button" className={`concept-card ${selected ? 'selected' : ''}`} aria-label={`${item.title}, concept card, filed ${new Date(item.created_at).toLocaleDateString()}`} aria-pressed={selected} onClick={onSelect}>
    <span className="card-index">{String.fromCharCode(65 + (item.id % 26))}—{String(item.id).padStart(3, '0')}</span>
    <strong>{item.title}</strong>
    <span className="ruled-lines">{item.note || 'Filed without a note.'}</span>
    <small>✦ ASTERISK CATALOG</small>
  </button>
}

function ConceptDetail({ item, onClose, onEdit, onRemove }: { item: Item; onClose: () => void; onEdit: (item: Item) => void; onRemove: (item: Item) => void }) {
  return <article className="concept-detail">
    <button type="button" className="detail-close" aria-label="Close concept details" onClick={onClose}>×</button>
    <div className="catalog-label"><small>SUBJECT</small><strong>{String.fromCharCode(65 + (item.id % 26))}</strong><span>DRAWER {String(item.id).padStart(2, '0')}</span></div>
    <div className="concept-detail-copy"><span>CONCEPT · FILED {new Date(item.created_at).toLocaleDateString()}</span><h3>{item.title}</h3>{item.note ? <p>{item.note}</p> : <p className="muted">No field note yet.</p>}{item.url && <a href={item.url} target="_blank" rel="noopener noreferrer">Open source ↗</a>}<div className="item-actions"><button onClick={() => onEdit(item)}>Edit this card</button><button onClick={() => void onRemove(item)}>Remove from catalog</button></div></div>
  </article>
}

function QuestionBulb({ item, selected, onSelect }: { item: Item; selected: boolean; onSelect: () => void }) {
  return <button type="button" className={`question-bulb ${selected ? 'selected' : ''}`} aria-label={`${item.title}, question lightbulb, added ${new Date(item.created_at).toLocaleDateString()}`} aria-pressed={selected} onClick={onSelect}>
    <span className="bulb-glass" aria-hidden="true"><i>?</i></span><span className="bulb-base" aria-hidden="true" />
    <strong>{item.title}</strong><small>QUESTION {String(item.id).padStart(2, '0')}</small>
  </button>
}

function QuestionDetail({ item, onClose, onEdit, onRemove }: { item: Item; onClose: () => void; onEdit: (item: Item) => void; onRemove: (item: Item) => void }) {
  return <article className="question-detail">
    <button type="button" className="detail-close" aria-label="Close question details" onClick={onClose}>×</button>
    <div className="inquiry-mark"><span>?</span><small>STILL<br />WONDERING</small></div>
    <div className="question-detail-copy"><span>QUESTION · ASKED {new Date(item.created_at).toLocaleDateString()}</span><h3>{item.title}</h3>{item.note ? <p>{item.note}</p> : <p className="muted">No clues gathered yet.</p>}{item.url && <a href={item.url} target="_blank" rel="noopener noreferrer">Follow the clue ↗</a>}<div className="item-actions"><button onClick={() => onEdit(item)}>Refine this question</button><button onClick={() => void onRemove(item)}>Remove from box</button></div></div>
  </article>
}

function ThoughtScrap({ item, selected, onSelect }: { item: Item; selected: boolean; onSelect: () => void }) {
  const scrapStyle = item.id % 3
  return <button type="button" className={`thought-scrap scrap-${scrapStyle} ${selected ? 'selected' : ''}`} aria-label={`${item.title}, random thought scrap, added ${new Date(item.created_at).toLocaleDateString()}`} aria-pressed={selected} onClick={onSelect}>
    <span className="scrap-pin" aria-hidden="true">✦</span><strong>{item.title}</strong>{item.note && <small>{item.note}</small>}<span className="scrap-date">{new Date(item.created_at).toLocaleDateString()}</span>
  </button>
}

function ThoughtDetail({ item, onClose, onEdit, onRemove }: { item: Item; onClose: () => void; onEdit: (item: Item) => void; onRemove: (item: Item) => void }) {
  return <article className="thought-detail">
    <button type="button" className="detail-close" aria-label="Close thought details" onClick={onClose}>×</button>
    <div className="thought-mark"><span>✦</span><strong>NOTE<br />TO SELF</strong><small>{new Date(item.created_at).toLocaleDateString()}</small></div>
    <div className="thought-detail-copy"><span>RANDOM THOUGHT · TUCKED AWAY</span><h3>{item.title}</h3>{item.note ? <p>{item.note}</p> : <p className="muted">The thought speaks for itself.</p>}{item.url && <a href={item.url} target="_blank" rel="noopener noreferrer">Follow this thread ↗</a>}<div className="item-actions"><button onClick={() => onEdit(item)}>Edit this thought</button><button onClick={() => void onRemove(item)}>Discard scrap</button></div></div>
  </article>
}

function ConceptExplorer({ item, onClose, onError, onChange }: { item: Item; onClose: () => void; onError: (message: string) => void; onChange: (count: number) => void }) {
  const [concepts, setConcepts] = useState<ItemConcept[]>([])
  const [loading, setLoading] = useState(true)
  const [extracting, setExtracting] = useState(false)
  const [name, setName] = useState('')
  const [category, setCategory] = useState<ItemConcept['category']>('personal')
  const [enrichment, setEnrichment] = useState<Enrichment | null>(null)
  const [enriching, setEnriching] = useState(false)
  const [identityHint, setIdentityHint] = useState('')

  async function load() {
    setLoading(true)
    try {
      const response = await fetch(`/api/items/${item.id}/concepts`)
      if (!response.ok) throw new Error(await responseError(response, 'Could not load concepts'))
      const loaded = await response.json()
      setConcepts(loaded)
      onChange(loaded.length)
      const enrichmentResponse = await fetch(`/api/items/${item.id}/enrichment`)
      if (enrichmentResponse.ok) setEnrichment(await enrichmentResponse.json())
    } catch (caught) { onError(caught instanceof Error ? caught.message : 'Could not load concepts') }
    finally { setLoading(false) }
  }

  useEffect(() => { void load() }, [item.id])

  async function extract() {
    setExtracting(true)
    onError('')
    try {
      const response = await fetch(`/api/items/${item.id}/concepts/extract`, { method: 'POST' })
      if (!response.ok) throw new Error(await responseError(response, 'Concept extraction failed'))
      const extracted = await response.json()
      setConcepts(extracted)
      onChange(extracted.length)
    } catch (caught) { onError(caught instanceof Error ? caught.message : 'Concept extraction failed') }
    finally { setExtracting(false) }
  }

  async function enrich(identityHintOverride = '') {
    setEnriching(true)
    onError('')
    try {
      const hint = identityHintOverride.trim()
      const response = await fetch(`/api/items/${item.id}/enrich`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ identity_hint: hint || null }) })
      if (!response.ok) throw new Error(await responseError(response, 'Research could not be completed'))
      const researched: Enrichment = await response.json()
      setEnrichment(researched)
      setConcepts(researched.concepts)
      onChange(researched.concepts.length)
    } catch (caught) { onError(caught instanceof Error ? caught.message : 'Research could not be completed') }
    finally { setEnriching(false) }
  }

  async function add(event: React.FormEvent) {
    event.preventDefault()
    if (!name.trim()) return
    const response = await fetch(`/api/items/${item.id}/concepts`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: name.trim(), category }) })
    if (!response.ok) { onError(await responseError(response, 'Could not add concept')); return }
    const created = await response.json()
    setConcepts(current => [...current, created])
    onChange(concepts.length + 1)
    setName('')
  }

  async function remove(concept: ItemConcept) {
    const response = await fetch(`/api/concepts/${concept.id}`, { method: 'DELETE' })
    if (!response.ok) { onError(await responseError(response, 'Could not remove concept')); return }
    setConcepts(current => current.filter(entry => entry.id !== concept.id))
    onChange(Math.max(0, concepts.length - 1))
  }

  return <section className="concept-workbench" id="concept-workbench">
    <button className="detail-close" type="button" aria-label="Close concept explorer" onClick={onClose}>×</button>
    <div className="concept-intro"><span>✦ RESEARCH & CONCEPTS</span><h2>{item.title}</h2><p>Research uses real web sources to identify this curiosity and find substantive ideas. Sources and AI interpretations stay separate from your original note. Full spoilers are allowed.</p></div>
    <div className="concept-actions"><button className="primary" type="button" disabled={enriching} onClick={() => void enrich()}>{enriching ? 'Researching the trail…' : enrichment?.status === 'researched' ? '↻ Refresh Research' : '✦ Enrich with Research'}</button><button className="secondary" type="button" disabled={extracting} onClick={() => void extract()}>{extracting ? 'Extracting…' : 'Quick extract without research'}</button></div>
    {enrichment && enrichment.status !== 'not_researched' && <section className={`research-card ${enrichment.status}`}><div className="research-status"><span>{enrichment.status === 'needs_confirmation' ? 'IDENTITY NEEDS CONFIRMATION' : 'RESEARCHED'}</span>{enrichment.researched_at && <time>{new Date(enrichment.researched_at).toLocaleString()}</time>}</div>{enrichment.status === 'needs_confirmation' ? <><p className="confirmation-question">{enrichment.confirmation_question}</p><div className="identity-confirm"><input aria-label="Identity details" placeholder="Author, director, year, or episode title" value={identityHint} onChange={event => setIdentityHint(event.target.value)} /><button type="button" disabled={!identityHint.trim() || enriching} onClick={() => void enrich(identityHint)}>Confirm & research</button></div></> : <><h3>{enrichment.resolved_title}</h3><p>{[enrichment.creator, enrichment.release_info].filter(Boolean).join(' · ')}</p><p>{enrichment.identity_summary}</p></>}<details><summary>{enrichment.sources.length} research {enrichment.sources.length === 1 ? 'source' : 'sources'}</summary><div className="research-sources">{enrichment.sources.map(source => <article key={source.id}><a href={source.url} target="_blank" rel="noopener noreferrer">{source.title} ↗</a><small>{source.publisher} · {source.retrieval_status}</small><p>{source.excerpt}</p></article>)}</div></details></section>}
    {enrichment && enrichment.rejected_leads.length > 0 && <details className="rejected-leads"><summary>{enrichment.rejected_leads.length} research {enrichment.rejected_leads.length === 1 ? 'lead' : 'leads'} rejected</summary>{enrichment.rejected_leads.map(lead => <p key={lead.name}><strong>{lead.name}</strong> — {lead.rejection_reason}</p>)}</details>}
    {loading ? <p className="muted">Opening the concept drawer…</p> : concepts.length === 0 ? <div className="concept-empty">No concepts yet. Ask Asterisk to explore, or add one in your own words.</div> : <div className="concept-list">{concepts.map(concept => <article className={`concept-chip ${concept.origin}`} key={concept.id}>
      <div><span>{concept.origin === 'ai' ? 'AI-INFERRED' : 'YOUR CONCEPT'} · {concept.level || concept.category}{concept.domain ? ` · ${concept.domain}` : ''}</span><strong>{concept.name}</strong>{(concept.definition || concept.explanation) && <p>{concept.definition || concept.explanation}</p>}{concept.mechanism && <p><b>How it works:</b> {concept.mechanism}</p>}{(concept.relationship_evidence || concept.relationship) && <p className="concept-relationship"><b>{concept.relationship_type || 'Relationship'}:</b> {concept.relationship_evidence || concept.relationship}</p>}{concept.concept_verification && <small>Concept: {concept.concept_verification} · Relationship: {concept.relationship_verification}</small>}</div>
      <button type="button" aria-label={`Remove ${concept.name}`} onClick={() => void remove(concept)}>×</button>
    </article>)}</div>}
    <form className="manual-concept" onSubmit={add}><label htmlFor="manual-concept-name">Add a concept for Sidequests</label><div><input id="manual-concept-name" maxLength={120} placeholder="A theme, principle, or underlying mechanism" value={name} onChange={event => setName(event.target.value)} /><select aria-label="Concept category" value={category} onChange={event => setCategory(event.target.value as ItemConcept['category'])}><option value="personal">Personal</option><option value="general">General theme</option><option value="mechanism">Mechanism</option></select><button disabled={!name.trim()}>Add concept</button></div></form>
  </section>
}

function SidequestsView({ items, conceptCounts, sidequests, busy, onExplore, onSurprise }: { items: Item[]; conceptCounts: Record<number, number>; sidequests: Sidequest[]; busy: boolean; onExplore: (item: Item) => void; onSurprise: () => void }) {
  const prepared = items.filter(item => (conceptCounts[item.id] || 0) > 0)
  const ready = prepared.length >= 2
  return <section className="sidequests" id="sidequests"><div className="sidequest-head"><div><span>03 / DISCOVER</span><h2>Sidequests</h2><p>Unexpected bridges between things that made you curious.</p></div><button className="surprise-button" type="button" disabled={busy || !ready} onClick={onSurprise}>{busy ? '✦ Following a thread…' : ready ? '✦ Surprise Me' : 'Prepare two curiosities'}</button></div>
    <div className="sidequest-setup">
      <div className="sidequest-explainer"><span>HOW IT WORKS</span><strong>Add concepts to any two curiosities.</strong><p>A concept is simply a theme or idea inside an item—not the “Concept” item type. The curiosities may be the same type; Sidequests uses their concepts as stepping stones and rejects connections that are too obvious.</p></div>
      <div className="sidequest-steps" aria-label="Sidequest readiness"><span className={items.length ? 'done' : ''}><b>1</b> Save curiosities</span><span className={ready ? 'done' : ''}><b>2</b> Prepare any two items</span><span className={ready ? 'done' : ''}><b>3</b> Ask to be surprised</span></div>
      {items.length === 0 ? <p className="sidequest-guidance">Save your first curiosity above to begin.</p> : <div className="sidequest-sources"><div className="sidequest-source-head"><strong>Choose curiosities to connect</strong><span>{prepared.length} of {items.length} prepared</span></div><div className="sidequest-source-list">{items.map(item => <article className={(conceptCounts[item.id] || 0) > 0 ? 'prepared' : ''} key={item.id}><span className="source-icon">{icons[item.kind]}</span><div><strong>{item.title}</strong><small>{labels[item.kind]} · {(conceptCounts[item.id] || 0) === 0 ? 'No concepts yet' : `${conceptCounts[item.id]} ${(conceptCounts[item.id] || 0) === 1 ? 'concept' : 'concepts'}`}</small></div><button type="button" onClick={() => onExplore(item)}>{(conceptCounts[item.id] || 0) > 0 ? 'View concepts' : 'Add concepts'}</button></article>)}</div></div>}
      {!ready && prepared.length > 0 && <p className="sidequest-guidance">Nice start. Add concepts to one more curiosity—of any type—to unlock Sidequests.</p>}
    </div>
    {sidequests.length === 0 ? <div className="sidequest-empty"><strong>{ready ? 'You’re ready for your first Sidequest.' : 'No Sidequests yet.'}</strong><p>{ready ? 'Your curiosities have enough stepping stones. Ask Asterisk to find an unexpected connection.' : 'Use the guide above to prepare two curiosities.'}</p></div> : <div className="sidequest-stack">{sidequests.map(sidequest => <article className="sidequest-card" key={sidequest.id}>
      <div className="evidence-label">✧ {sidequest.evidence_status}</div><h3>{sidequest.title}</h3><div className="connected-items"><span>{sidequest.item_a_title}</span><b>✦</b><span>{sidequest.item_b_title}</span></div>
      <div className="sidequest-section"><h4>The unexpected connection</h4><p>{sidequest.connection}</p></div>
      <div className="sidequest-grid"><div><h4>The bridge</h4><p>{sidequest.bridge}</p></div><div><h4>Why it’s interesting</h4><p>{sidequest.why_interesting}</p></div><div><h4>Where it breaks</h4><p>{sidequest.where_it_breaks}</p></div><div className="new-question"><h4>A new question</h4><p>{sidequest.new_question}</p></div></div>
      {(sidequest.concepts_used.length > 0 || sidequest.evidence.length > 0) && <details className="sidequest-evidence"><summary>Inspect concepts & evidence</summary>{sidequest.concepts_used.length > 0 && <p><strong>Concepts:</strong> {sidequest.concepts_used.join(' · ')}</p>}{sidequest.evidence.length > 0 && <ul>{sidequest.evidence.map(reference => <li key={reference}><a href={reference} target="_blank" rel="noopener noreferrer">{reference}</a></li>)}</ul>}</details>}
      <time>{new Date(sidequest.created_at).toLocaleDateString()}</time>
    </article>)}</div>}
  </section>
}

function CollectionEmpty({ icon, message, action, onAction }: { icon: string; message: string; action: string; onAction: () => void }) {
  return <div className="empty collection-empty"><span>{icon}</span><strong>{message}</strong><button type="button" onClick={onAction}>{action}</button></div>
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>)
