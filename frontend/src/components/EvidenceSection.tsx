import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import type { EvidenceReference } from '../api/types'
import './EvidenceSection.css'

interface EvidenceSectionProps {
  evidence: EvidenceReference[]
}

function formatTimestamp(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
  const secs = Math.round(seconds % 60)
  return `${minutes}:${secs.toString().padStart(2, '0')}`
}

/** Builds the short "filename · p.4" / "lecture.mp3 · 03:12-03:28" locator
 * line, using ONLY fields the backend actually populated -- never a
 * fabricated page/timestamp (see backend/agents/provenance.py). */
function locatorLine(ref: EvidenceReference): string {
  const label = ref.source === 'youtube_transcript' ? 'video transcript' : (ref.filename ?? 'source')
  const parts = [label]

  if (ref.page != null) {
    parts.push(`p. ${ref.page}`)
  } else if (ref.segment_start_seconds != null && ref.segment_end_seconds != null) {
    parts.push(`${formatTimestamp(ref.segment_start_seconds)}–${formatTimestamp(ref.segment_end_seconds)}`)
  } else if (ref.segment_start_seconds != null) {
    parts.push(formatTimestamp(ref.segment_start_seconds))
  }

  return parts.join(' · ')
}

/** Compact, expandable "Sources" section -- one entry per EvidenceReference
 * the backend actually attached to this answer. Never shows a retrieval
 * score unless it's genuinely present (RAG evidence only). */
export function EvidenceSection({ evidence }: EvidenceSectionProps) {
  const [isOpen, setIsOpen] = useState(false)
  if (evidence.length === 0) return null

  return (
    <div className="evidence-section">
      <button
        type="button"
        className="evidence-section__toggle"
        onClick={() => setIsOpen((open) => !open)}
        aria-expanded={isOpen}
      >
        <span className="evidence-section__toggle-title">Sources</span>
        <span className="evidence-section__toggle-meta">{evidence.length}</span>
        <ChevronDown size={14} className={isOpen ? 'evidence-section__chevron--open' : ''} />
      </button>

      {isOpen && (
        <ol className="evidence-section__list">
          {evidence.map((ref, index) => (
            <li key={index} className="evidence-section__item">
              <span className="evidence-section__locator">
                {locatorLine(ref)}
                {ref.score != null && <span className="evidence-section__score"> · {(ref.score * 100).toFixed(0)}% match</span>}
              </span>
              <p className="evidence-section__excerpt">"{ref.excerpt}"</p>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
