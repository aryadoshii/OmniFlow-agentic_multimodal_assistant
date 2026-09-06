import { GitCompare } from 'lucide-react'
import type { CrossSourceAnalysis, CrossSourceRelationship } from '../api/types'
import './CrossSourceAnalysisView.css'

interface CrossSourceAnalysisViewProps {
  analysis: CrossSourceAnalysis
}

const RELATIONSHIP_LABEL: Record<CrossSourceRelationship, string> = {
  strong_overlap: 'Strong overlap',
  partial_overlap: 'Partial overlap',
  different_topics: 'Different topics',
  contradiction: 'Contradiction',
  insufficient_evidence: 'Insufficient evidence',
}

const RELATIONSHIP_TONE: Record<CrossSourceRelationship, 'success' | 'warning' | 'neutral' | 'error'> = {
  strong_overlap: 'success',
  partial_overlap: 'warning',
  different_topics: 'neutral',
  contradiction: 'error',
  insufficient_evidence: 'neutral',
}

/** Compact cross-source (multi-document) consistency/relationship panel --
 * only ever rendered from the backend's own structured CrossSourceAnalysis
 * (backend/agents/cross_source.py), never fabricated client-side. */
export function CrossSourceAnalysisView({ analysis }: CrossSourceAnalysisViewProps) {
  const tone = RELATIONSHIP_TONE[analysis.relationship]

  return (
    <div className="cross-source">
      <div className="cross-source__header">
        <span className="cross-source__label">
          <GitCompare size={13} /> Cross-source analysis
        </span>
        <span className={`cross-source__badge cross-source__badge--${tone}`}>
          {RELATIONSHIP_LABEL[analysis.relationship]}
        </span>
      </div>

      {analysis.sources.length > 0 && (
        <div className="cross-source__sources">
          {analysis.sources.map((source) => (
            <div key={source.document_id} className="cross-source__source">
              <span className="cross-source__source-filename">{source.filename}</span>
              <span className="cross-source__source-arrow">→</span>
              <span className="cross-source__source-topic">{source.topic}</span>
            </div>
          ))}
        </div>
      )}

      {analysis.shared_concepts.length > 0 && (
        <div className="cross-source__row">
          <span className="cross-source__row-label">Shared</span>
          <span className="cross-source__row-value">{analysis.shared_concepts.join(', ')}</span>
        </div>
      )}

      {analysis.differences.length > 0 && (
        <div className="cross-source__row">
          <span className="cross-source__row-label">Differences</span>
          <ul className="cross-source__differences">
            {analysis.differences.map((difference, index) => (
              <li key={index}>{difference}</li>
            ))}
          </ul>
        </div>
      )}

      <p className="cross-source__explanation">{analysis.explanation}</p>
    </div>
  )
}
