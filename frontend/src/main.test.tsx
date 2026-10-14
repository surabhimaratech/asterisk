import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App, ConceptExplorer, RejectedFindingCard } from './main'

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
