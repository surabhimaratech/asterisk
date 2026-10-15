import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App, ConceptExplorer, DiscoveryCard, ItemCard, RejectedFindingCard, SidequestsView } from './main'

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
    expect(screen.getAllByText(item.title)).toHaveLength(2)
    expect(screen.queryByText('✦', { selector: '.connected-items b' })).not.toBeInTheDocument()
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
