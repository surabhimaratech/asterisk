import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App, ConceptExplorer, DiscoveryCard, ItemCard, RejectedFindingCard, ResearchSidequestBeta, SidequestsView } from './main'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  localStorage.clear()
})

describe('media types', () => {
  it('offers TV Series separately from Movie', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      return new Response(JSON.stringify(url.includes('/sidequests') ? [] : []), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }))
    render(<App />)
    expect((await screen.findAllByRole('button', { name: /TV Series/i })).length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: /^🎬 Movie$/i })).toBeInTheDocument()
  })

  it('offers accessible work scope, progress, and spoiler controls', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify([]), { status: 200, headers: { 'Content-Type': 'application/json' } })))
    render(<App />)

    fireEvent.click(screen.getByText('Identity, progress & spoilers'))
    expect(screen.getByLabelText('What does this entry represent?')).toHaveValue('unknown')
    fireEvent.change(screen.getByLabelText('What does this entry represent?'), { target: { value: 'series' } })
    expect(screen.getByLabelText('Your progress')).toBeInTheDocument()
    expect(screen.getByLabelText('Spoiler boundary')).toHaveValue('avoid')
  })
})

describe('curiosity index cards', () => {
  const item = {
    id: 21, kind: 'book' as const, title: 'Rendezvous with Rama', note: 'A mysterious visitor.', url: 'https://example.com/rama',
    creator: 'Arthur C. Clarke', release_year: 1973, work_scope: 'individual' as const, parent_work_title: null,
    canonical_identity_id: null, progress_status: 'finished' as const, progress_current: null, progress_total: null,
    progress_unit: 'book' as const, spoiler_policy: 'full' as const, created_at: new Date().toISOString(),
  }

  it('opens concepts from the card while keeping its controls independent', () => {
    const onExplore = vi.fn()
    const onEdit = vi.fn()
    render(<ItemCard item={item} conceptCount={0} onExplore={onExplore} onEdit={onEdit} onRemove={async () => undefined} />)

    fireEvent.click(screen.getByRole('article'))
    expect(onExplore).toHaveBeenCalledOnce()
    fireEvent.click(screen.getByRole('button', { name: 'Edit' }))
    expect(onEdit).toHaveBeenCalledOnce()
    expect(onExplore).toHaveBeenCalledOnce()
  })

  it('labels previously extracted concepts as viewable', () => {
    render(<ItemCard item={item} conceptCount={3} onExplore={() => undefined} onEdit={() => undefined} onRemove={async () => undefined} />)
    expect(screen.getByRole('button', { name: 'View Concepts' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Explore Concepts' })).not.toBeInTheDocument()
  })
})

describe('human review controls', () => {
  it('keeps interest separate from evidence and keep decisions', async () => {
    const onInterest = vi.fn(async () => undefined)
    const onReview = vi.fn(async () => undefined)
    const finding = {
      id: 7, item_id: 2, item_title: 'Cytonic', item_kind: 'book' as const, name: 'Identity Fragmentation', stage: 'concept_candidate' as const,
      proposed_definition: 'A grounded synthesis.', motivating_detail: '', proposed_relationship: 'Memory shapes continuity.', relationship_type: 'interpretive',
      rejection_reason: 'The relationship needs more evidence.', discovery_origin: 'work_grounded' as const, research_path: 'A traceable research path.', research_questions: ['memory continuity'],
      initial_validation_outcome: 'Quote failed.', recovery_attempt: {}, final_validation_outcome: 'Still unverified.', automated_evidence_status: 'unverified' as const,
      human_review_status: 'pending' as const, human_review_note: null, review_history: [], kept_concept_id: null, evidence_review_detail: null, last_evaluated_at: null,
      candidate: {}, validation: {}, sources: [], superseded_at: null, concept_classification: 'grounded_interpretation' as const, evidence_form: 'paraphrase' as const,
      feedback_key: 'stable-key', discovery_mode: 'deepen_fascination' as const, discovery_quality: { specificity: 'high', intellectual_substance: 'high', surprise_potential: 'medium' },
      interested: false, interest_note: null, interested_at: null, loved: false, love_note: null, loved_at: null, surprised: false, surprise_note: null, surprised_at: null, created_at: new Date().toISOString(),
    }
    const onFeedback = vi.fn(async () => undefined)
    render(<RejectedFindingCard finding={finding} onReview={onReview} onReevaluate={async () => undefined} onInterest={onInterest} onFeedback={onFeedback} />)
    expect(screen.getByText(/GROUNDED INTERPRETATION/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '☆ Interested' }))
    await waitFor(() => expect(onInterest).toHaveBeenCalledOnce())
    expect(onReview).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: '♡ Love' }))
    await waitFor(() => expect(onFeedback).toHaveBeenCalledWith(finding, 'love'))
    fireEvent.click(screen.getByRole('button', { name: 'Keep anyway' }))
    await waitFor(() => expect(onReview).toHaveBeenCalledWith(finding, 'keep', ''))
  })
})

describe('discovery card hierarchy', () => {
  it('keeps the invitation visible, research collapsed, and reactions accessible', async () => {
    const concept = {
      id: 12, name: 'Humor as investigative friction', origin: 'ai',
      definition: 'Playful speculation collides with procedural skepticism during a case.',
      explanation: null, relationship_evidence: 'A retrieved episode account supports the interaction.', relationship: null,
      mechanism: 'Competing tones become a practical way to test explanations without flattening the relationship.',
      curiosity_hook: 'When can humor help people test competing explanations instead of distracting from them?',
      user_note: null, evidence_status: 'work_supported', discovery_origin: 'work_grounded',
      concept_classification: 'grounded_observation', level: 'mechanism', category: 'mechanism', domain: 'tone or humor',
      loved: false, surprised: false, interested: false, interest_note: null, love_note: null, surprise_note: null,
      source_support: [], appeal_dimensions: ['funny', 'clever'], pattern_scope: 'single_work', spoiler_scope: 'premise_only',
      recovery_attempt: {}, relationship_type: 'explicit', research_path: 'Specific to the work.',
      concept_verification: 'work_supported', relationship_verification: 'verified', discovery_quality: { surprise_potential: 'medium' },
    }
    const onFeedback = vi.fn(async () => undefined)
    const onInterest = vi.fn(async () => undefined)
    const onExploration = vi.fn(async () => undefined)
    render(<DiscoveryCard concept={concept as never} enrichment={null} onRemove={async () => undefined} onInterest={onInterest} onFeedback={onFeedback} onExploration={onExploration} />)

    expect(screen.getByText('Where could this lead?')).toBeInTheDocument()
    expect(screen.getByText(concept.curiosity_hook)).toBeInTheDocument()
    expect(screen.getByText('Research & provenance').closest('details')).not.toHaveAttribute('open')
    expect(screen.getByRole('button', { name: `Love ${concept.name}` })).toHaveAttribute('aria-pressed', 'false')
    fireEvent.click(screen.getByRole('button', { name: `Save for a Sidequest: ${concept.curiosity_hook}` }))
    await waitFor(() => expect(onExploration).toHaveBeenCalledWith(concept, undefined))
    fireEvent.click(screen.getByText('Add context to your reaction'))
    fireEvent.change(screen.getByPlaceholderText(/what caught you/i), { target: { value: 'I already knew this, but I still love it.' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save with Love' }))
    await waitFor(() => expect(onFeedback).toHaveBeenCalledWith(concept, 'love', 'I already knew this, but I still love it.', true))
  })

  it('renders a short personal concept without empty research scaffolding', () => {
    const concept = { id: 13, name: 'Mystical cafés', origin: 'user', explanation: null, definition: null, relationship_evidence: null, relationship: null, mechanism: null, curiosity_hook: null, user_note: null, loved: false, surprised: false, interested: false, interest_note: null, love_note: null, surprise_note: null, source_support: [], appeal_dimensions: [], pattern_scope: null, spoiler_scope: 'premise_only', recovery_attempt: {}, discovery_origin: null, concept_classification: null, level: null, category: 'personal', domain: null, evidence_status: null, relationship_type: null, research_path: null, concept_verification: null, relationship_verification: null, discovery_quality: {} }
    render(<DiscoveryCard concept={concept as never} enrichment={null} onRemove={async () => undefined} onInterest={async () => undefined} onFeedback={async () => undefined} />)
    expect(screen.getByRole('heading', { name: 'Mystical cafés' })).toBeInTheDocument()
    expect(screen.queryByText('Research & provenance')).not.toBeInTheDocument()
    expect(screen.queryByText('Where could this lead?')).not.toBeInTheDocument()
  })
})

describe('Sidequest preparation', () => {
  it('shows a no-case outcome without pretending a Sidequest was created', () => {
    const item = { id: 1, kind: 'book' as const, title: 'A Psalm for the Wild-Built', note: null, url: null, creator: null, release_year: null, work_scope: 'individual' as const, parent_work_title: null, canonical_identity_id: null, progress_status: 'finished' as const, progress_current: null, progress_total: null, progress_unit: 'book' as const, spoiler_policy: 'full' as const, created_at: new Date().toISOString() }
    render(<SidequestsView items={[item]} conceptCounts={{ 1: 1 }} sidequests={[]} invitations={[]} busy={false} outcome="Asterisk investigated the available leads but did not save a Sidequest. The critic found only a thematic parallel. No compelling case yet. Asterisk left the investigation open rather than saving a weak connection." onExplore={() => undefined} onSurprise={() => undefined} />)

    expect(screen.getByRole('status')).toHaveTextContent('No compelling case yet.')
    expect(screen.getByRole('status')).toHaveTextContent('The critic found only a thematic parallel.')
    expect(screen.getByRole('status')).toHaveTextContent('left the investigation open rather than saving a weak connection')
    expect(screen.queryByText(/new Sidequest sparked/i)).not.toBeInTheDocument()
  })

  it('shows saved questions as invitations rather than completed investigations', () => {
    const base = { note: null, url: null, creator: null, release_year: null, work_scope: 'individual' as const, parent_work_title: null, canonical_identity_id: null, progress_status: 'finished' as const, progress_current: null, progress_total: null, progress_unit: 'book' as const, spoiler_policy: 'full' as const, created_at: new Date().toISOString() }
    const items = [{ ...base, id: 1, kind: 'book' as const, title: 'First' }, { ...base, id: 2, kind: 'book' as const, title: 'Second' }]
    const invitations = [{ id: 8, item_id: 1, item_title: 'First', source_concept_id: 3, active_concept_id: 3, source_concept_name: 'Shared sensory language', question: 'How can incompatible senses still support a shared language?', status: 'queued' as const, user_note: null, available: true, created_at: base.created_at, updated_at: base.created_at }]

    render(<SidequestsView items={items} conceptCounts={{ 1: 1, 2: 1 }} sidequests={[]} invitations={invitations} busy={false} onExplore={() => undefined} onSurprise={() => undefined} />)

    expect(screen.getByText('Questions you want to follow')).toBeInTheDocument()
    expect(screen.getByText(invitations[0].question)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '✦ Discover Sidequests' })).toBeEnabled()
    expect(screen.queryByText(/completed investigation/i)).not.toBeInTheDocument()
  })

  it('renders a single-origin Sidequest without inventing a second curiosity', () => {
    const created = new Date().toISOString()
    const item = { id: 1, kind: 'thought' as const, title: 'Maps of hesitation', note: null, url: null, creator: null, release_year: null, work_scope: 'individual' as const, parent_work_title: null, canonical_identity_id: null, progress_status: 'unknown' as const, progress_current: null, progress_total: null, progress_unit: 'installment' as const, spoiler_policy: 'avoid' as const, created_at: created }
    const sidequest = { id: 4, item_a_id: 1, item_b_id: 1, item_a_title: item.title, item_b_title: item.title, title: 'A map of almost-chosen paths', connection: 'Hesitation can become useful spatial data.', bridge: 'Spatial traces of unrealized decisions', why_interesting: 'It preserves uncertainty.', where_it_breaks: 'Pauses can also be noise or access barriers.', new_question: 'What do abandoned routes reveal?', evidence_status: 'Exploratory and not externally verified.', concepts_used: ['Hesitation as spatial data'], evidence: [], origin_item_ids: [1], origin_titles: [item.title], discovery_kind: 'creative_impulse', invitation: 'Trace where uncertainty changed your route.', created_at: created }

    render(<SidequestsView items={[item]} conceptCounts={{ 1: 1 }} sidequests={[sidequest]} invitations={[]} busy={false} onExplore={() => undefined} onSurprise={() => undefined} />)

    expect(screen.getByText('Trace where uncertainty changed your route.')).toBeInTheDocument()
    expect(screen.getByText('The unexpected connection')).toBeInTheDocument()
    expect(screen.queryByText('DETECTIVE MODE', { exact: false })).not.toBeInTheDocument()
    expect(screen.getAllByText(item.title)).toHaveLength(2)
    expect(screen.queryByText('✦', { selector: '.connected-items b' })).not.toBeInTheDocument()
  })

  it('renders Detective Mode as a readable case file with distinct claim states', () => {
    const created = new Date().toISOString()
    const item = { id: 1, kind: 'book' as const, title: 'Piranesi', note: null, url: null, creator: null, release_year: null, work_scope: 'individual' as const, parent_work_title: null, canonical_identity_id: null, progress_status: 'finished' as const, progress_current: null, progress_total: null, progress_unit: 'book' as const, spoiler_policy: 'full' as const, created_at: created }
    const sidequest = {
      id: 9, item_a_id: 1, item_b_id: 1, item_a_title: item.title, item_b_title: item.title,
      title: 'The map that remembers uncertainty', connection: 'External memory can make unstable patterns navigable.',
      bridge: 'External memory for vanishing patterns', why_interesting: 'A survival record becomes a map of possibility.',
      where_it_breaks: 'Tides recur independently, while hesitation is subjective.', new_question: 'What should a map remember besides movement?',
      evidence_status: 'Exploratory · sourced details supported; interpretations and open leads unverified.',
      concepts_used: ['Tides as external memory'], evidence: ['https://example.com/piranesi'], origin_item_ids: [1], origin_titles: [item.title],
      discovery_kind: 'new_lens', invitation: 'Look for the moment a possibility disappears.', detective_version: 1,
      clue: { text: "Piranesi's tidal journals make recurring floods navigable.", item_id: 1, concept_names: ['Tides as external memory'], source_ids: ['source-1'] },
      lead: { text: 'Counterfactual cartography could preserve paths not taken.', kind: 'unfamiliar_subject' as const, category: 'artistic_technique' as const, source_ids: [] },
      reasoning_steps: ['The journal records recurrence outside memory.', 'A hesitation map could preserve a choice before it disappears.'],
      claims: [
        { claim_type: 'supported_fact' as const, text: 'The journals record recurring tides.', source_ids: ['source-1'] },
        { claim_type: 'interpretation' as const, text: 'The journal acts as external memory.', source_ids: [] },
        { claim_type: 'speculative_hypothesis' as const, text: 'Maps might preserve unrealized decisions.', source_ids: [] },
      ], created_at: created,
    }

    render(<SidequestsView items={[item]} conceptCounts={{ 1: 1 }} sidequests={[sidequest]} invitations={[]} busy={false} onExplore={() => undefined} onSurprise={() => undefined} />)

    expect(screen.getByText(/DETECTIVE MODE/)).toBeInTheDocument()
    expect(screen.getByText('The clue')).toBeInTheDocument()
    expect(screen.getByText('The unexpected lead')).toBeInTheDocument()
    expect(screen.getByText(/artistic technique · Open lead/)).toBeInTheDocument()
    expect(screen.getByText('How the trail connects')).toBeInTheDocument()
    expect(screen.getByText('What’s supported')).toBeInTheDocument()
    expect(screen.getByText('Asterisk’s interpretation')).toBeInTheDocument()
    expect(screen.getByText('The hunch')).toBeInTheDocument()
    expect(screen.getByText('Next investigation')).toBeInTheDocument()
  })
})

describe('Discovery Engine beta', () => {
  it('explores, dismisses, saves, and reopens discoveries without repeating research', async () => {
    const item = { id: 9, kind: 'book' as const, title: 'Piranesi', note: 'Tides and journals', url: null, creator: null, release_year: null, work_scope: 'individual' as const, parent_work_title: null, canonical_identity_id: null, progress_status: 'finished' as const, progress_current: null, progress_total: null, progress_unit: 'book' as const, spoiler_policy: 'full' as const, created_at: new Date().toISOString() }
    const result = {
      seed_item_id: 9, seed_title: 'Piranesi', directions: [], suggested_candidate_id: 'tides',
      discovery_candidates: [
        { candidate_id: 'tides', hook: 'When tide tables became port infrastructure', story: 'Observers published recurring water levels as tables. Harbor workers could then coordinate around a prediction shared by strangers.', twist: 'A private notebook practice became public infrastructure.', library_connection: null, research_direction: 'How did tide tables change ports?', originating_curiosity: 'Journals record recurring tides.', evidence_assessment: 'The archive directly supports publication and use.', go_deeper: 'Which port published the first public table?', ai_suggested: true, evidence: [{ source_id: 'discovery-source-1', url: 'https://example.org/tides', title: 'Tide archive', excerpt: 'Ports published tables for navigation.' }] },
        { candidate_id: 'flood-mark', hook: 'The flood a building refused to forget', story: 'A flood mark was retained when a building was renovated. The rebuilt surface still carries the height of one city-changing event.', twist: 'Renovation preserved damage as a public warning instead of erasing it.', library_connection: 'Piranesi also turns architecture into a keeper of memory.', research_direction: 'Which buildings encode water levels?', originating_curiosity: 'The halls remember recurring water.', evidence_assessment: 'The city archive identifies the mark and flood.', go_deeper: 'Who decided to preserve the mark?', ai_suggested: false, evidence: [{ source_id: 'discovery-source-2', url: 'https://example.org/flood', title: 'City flood archive', excerpt: 'The flood mark was retained during renovation.' }] },
      ], title: 'When tide tables became infrastructure',
      discovery: 'Published observations helped ports predict recurring water levels.', why_surprising: 'A notebook-like practice coordinated strangers.',
      connection_to_library: null, what_remains_uncertain: 'Adoption varied by port.', next_rabbit_hole: 'Which port published the first public table?',
      claims: [{ claim_type: 'supported_fact', text: 'Ports published tide tables.', source_ids: ['discovery-source-1'] }],
      evidence: [{ source_id: 'discovery-source-1', url: 'https://example.org/tides', title: 'Tide archive', excerpt: 'Ports published tables for navigation.' }],
      evidence_status: 'Research-backed', persisted: false,
    }
    let stored: unknown[] = []
    let failSave = true
    let failRemove = true
    const fetchMock = vi.fn(async (url: string, options?: RequestInit) => {
      if (url === '/api/discoveries/research') return Response.json(result)
      if (options?.method === 'POST') {
        if (failSave) return new Response(null, { status: 503 })
        stored = JSON.parse(String(options.body))
      }
      if (options?.method === 'DELETE') {
        if (failRemove) return new Response(null, { status: 503 })
        stored = []
        return new Response(null, { status: 204 })
      }
      return Response.json(stored)
    })
    vi.stubGlobal('fetch', fetchMock)
    const { unmount } = render(<ResearchSidequestBeta items={[item]} />)

    fireEvent.change(screen.getByLabelText('Start with'), { target: { value: '9' } })
    fireEvent.click(screen.getByRole('button', { name: '✦ Research a Sidequest' }))

    expect(await screen.findByText('When tide tables became infrastructure')).toBeInTheDocument()
    expect(screen.getAllByRole('link', { name: 'Tide archive ↗' })[0]).toHaveAttribute('href', 'https://example.org/tides')
    expect(screen.getByText('✦ ASTERISK’S SUGGESTED PICK')).toBeInTheDocument()
    const floodCard = screen.getByText('The flood a building refused to forget').closest('article')!
    fireEvent.click(within(floodCard).getByRole('button', { name: 'Explore' }))
    expect(within(floodCard).getByText('Who decided to preserve the mark?')).toBeInTheDocument()
    expect(within(floodCard).getByText('Piranesi also turns architecture into a keeper of memory.')).toBeInTheDocument()
    expect(within(floodCard).getByText('Which buildings encode water levels?')).not.toBeVisible()
    fireEvent.click(within(floodCard).getByText('Sources & details'))
    expect(within(floodCard).getByText('Which buildings encode water levels?')).toBeVisible()
    expect(within(floodCard).getAllByRole('link', { name: 'City flood archive ↗' })[0]).toHaveAttribute('href', 'https://example.org/flood')
    const save = within(floodCard).getByRole('button', { name: 'Save for later' })
    fireEvent.click(save)
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not save the discovery')
    expect(screen.queryByRole('complementary', { name: 'Saved discoveries' })).not.toBeInTheDocument()
    failSave = false
    fireEvent.click(save)
    expect(await screen.findByRole('complementary', { name: 'Saved discoveries' })).toBeInTheDocument()
    const tideCard = screen.getByText('When tide tables became port infrastructure').closest('article')!
    fireEvent.click(within(floodCard).getByRole('button', { name: 'Not interested' }))
    expect(screen.getByText('1 of 2 showing')).toBeInTheDocument()
    fireEvent.click(within(tideCard).getByRole('button', { name: 'Not interested' }))
    expect(screen.getByText('All findings hidden for this run.')).toBeInTheDocument()
    expect(screen.queryByText('When tide tables became infrastructure')).not.toBeInTheDocument()
    expect(screen.queryByText('Connection to my library')).not.toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith('/api/discoveries/research', expect.objectContaining({
      method: 'POST', body: JSON.stringify({ item_id: 9 }),
    }))

    unmount()
    const restored = render(<ResearchSidequestBeta items={[item]} />)
    const saved = await screen.findByRole('complementary', { name: 'Saved discoveries' })
    const requestsBeforeOpen = fetchMock.mock.calls.length
    expect(within(saved).queryByText('When tide tables became port infrastructure')).not.toBeInTheDocument()
    const summary = within(saved).getByText('The flood a building refused to forget')
    expect(within(saved).getByText('Who decided to preserve the mark?')).not.toBeVisible()
    fireEvent.click(summary)
    expect(within(saved).getByText('Who decided to preserve the mark?')).toBeVisible()
    expect(within(saved).getByText(result.discovery_candidates[1].story)).toBeVisible()
    expect(within(saved).getByRole('link', { name: 'City flood archive ↗' })).toHaveAttribute('href', 'https://example.org/flood')
    expect(fetchMock).toHaveBeenCalledTimes(requestsBeforeOpen)
    const remove = within(saved).getByRole('button', { name: /Remove saved discovery/ })
    fireEvent.click(remove)
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not remove the discovery')
    expect(saved).toBeInTheDocument()
    failRemove = false
    fireEvent.click(remove)
    await waitFor(() => expect(saved).not.toBeInTheDocument())
    restored.unmount()
    render(<ResearchSidequestBeta items={[item]} />)
    await waitFor(() => expect(fetchMock.mock.calls.filter(([url, options]) =>
      url === '/api/discoveries/saved' && !options?.method)).toHaveLength(3))
    expect(screen.queryByRole('complementary', { name: 'Saved discoveries' })).not.toBeInTheDocument()
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/discoveries/research')).toHaveLength(1)
  })

  it('retries a failed browser import and clears the browser copy after confirmation', async () => {
    const entry = {
      candidate_id: 'flood-mark', hook: 'The flood a building refused to forget',
      story: 'The rebuilt surface still carries the height of the flood.',
      go_deeper: 'Who decided to preserve the mark?',
      evidence: [{ source_id: 'archive', url: 'https://example.org/flood', title: 'Archive', excerpt: 'The mark was retained.' }],
    }
    const key = 'asterisk.saved-discoveries.v1'
    const legacy = JSON.stringify([
      { ...entry, extra: true },
      { ...entry, candidate_id: 'bad', evidence: [{ ...entry.evidence[0], url: 'https://' }] },
      null,
    ])
    localStorage.setItem(key, legacy)
    let failImport = true
    const fetchMock = vi.fn(async (_url: string, options?: RequestInit) => {
      if (options?.method !== 'POST') return Response.json([])
      return failImport ? new Response(null, { status: 503 }) : Response.json([entry])
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<ResearchSidequestBeta items={[]} />)
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not finish importing browser saves')
    expect(localStorage.getItem(key)).toBe(legacy)
    expect(fetchMock.mock.calls[0][0]).toBe('/api/discoveries/saved')
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual([entry])
    failImport = false
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(await screen.findByRole('complementary', { name: 'Saved discoveries' })).toBeInTheDocument()
    expect(localStorage.getItem(key)).toBeNull()
  })

  it.each(['malformed', 'inaccessible'])('loads SQLite saves when browser data is %s', async problem => {
    if (problem === 'malformed') localStorage.setItem('asterisk.saved-discoveries.v1', '{broken')
    else vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('Storage access denied', 'SecurityError')
    })
    vi.stubGlobal('fetch', vi.fn(async () => Response.json([{
      candidate_id: 'existing', hook: 'Already in SQLite', story: 'A saved story',
      go_deeper: 'What happened next?', evidence: [],
    }])))

    render(<ResearchSidequestBeta items={[]} />)
    expect(await screen.findByText('Already in SQLite')).toBeInTheDocument()
  })
})

describe('identity disambiguation', () => {
  it('shows inspectable candidates and confirms the selected work', async () => {
    const ambiguous = {
      status: 'needs_confirmation', resolved_title: null, creator: null, release_info: null, identity_summary: null,
      confirmation_question: 'Choose the intended series.', identity_status: 'ambiguous', identity_key: null,
      resolved_media_type: null, identity_confidence: null, confirmed_at: null, researched_at: null, sources: [], concepts: [], rejected_leads: [],
      identity_candidates: [{ candidate_id: 'castle-2009', title: 'Castle', creator: 'Andrew W. Marlowe', release_year: 2009, media_type: 'tv_series', description: 'The American crime comedy-drama series.', confidence: 0.98, source_ids: ['identity-1'] }],
    }
    const confirmed = { ...ambiguous, status: 'not_researched', identity_status: 'confirmed', identity_key: 'tv_series|castle|andrew w marlowe|2009', resolved_title: 'Castle', creator: 'Andrew W. Marlowe', release_info: '2009', resolved_media_type: 'tv_series', identity_summary: ambiguous.identity_candidates[0].description, confirmed_at: new Date().toISOString(), identity_candidates: ambiguous.identity_candidates }
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      const body = url.endsWith('/concepts') ? [] : url.endsWith('/identity/confirm') ? confirmed : ambiguous
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    const item = {
      id: 3, kind: 'tv_series' as const, title: 'Castle', note: null, url: null,
      creator: null, release_year: null, work_scope: 'individual' as const,
      parent_work_title: null, canonical_identity_id: null, progress_status: 'unknown' as const,
      progress_current: null, progress_total: null, progress_unit: 'installment' as const,
      spoiler_policy: 'avoid' as const, created_at: new Date().toISOString(),
    }
    render(<ConceptExplorer item={item} onClose={() => undefined} onError={() => undefined} onChange={() => undefined} />)
    expect(await screen.findByText('Andrew W. Marlowe · 2009 · TV Series')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'This is the one' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/items/3/identity/confirm', expect.objectContaining({ method: 'POST' })))
    expect(await screen.findByText('CONFIRMED IDENTITY')).toBeInTheDocument()
  })
})
