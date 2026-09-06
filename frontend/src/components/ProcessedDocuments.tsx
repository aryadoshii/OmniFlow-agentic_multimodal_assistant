import { useState } from 'react'
import { ChevronDown, FileAudio, FileImage, FileText } from 'lucide-react'
import type { NormalizedDocument } from '../api/types'
import './ProcessedDocuments.css'

interface ProcessedDocumentsProps {
  documents: NormalizedDocument[]
}

const SOURCE_ICON = {
  pdf: FileText,
  text: FileText,
  image: FileImage,
  audio: FileAudio,
} as const

const EXTRACTION_LABEL: Record<string, string> = {
  direct_input: 'Direct input',
  native_text: 'Native text',
  ocr: 'OCR',
  mixed: 'Native text + OCR',
  speech_to_text: 'Speech-to-text',
}

const SOURCE_TYPE_LABEL: Record<string, string> = {
  pdf: 'PDF',
  text: 'Text',
  image: 'Image',
  audio: 'Audio',
}

function formatDuration(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.round(totalSeconds % 60)
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

/** Builds a short, factual summary line strictly from fields the backend
 * actually returns (see omniflow/models/document.py) -- never invented. */
function summaryLine(doc: NormalizedDocument): string {
  const typeLabel = SOURCE_TYPE_LABEL[doc.source_type] ?? doc.source_type
  const meta = doc.metadata ?? {}

  if (doc.source_type === 'audio' && typeof meta.duration_seconds === 'number') {
    return `${typeLabel} · ${formatDuration(meta.duration_seconds)}`
  }

  const parts: string[] = [typeLabel, EXTRACTION_LABEL[doc.extraction_method] ?? doc.extraction_method]

  if (doc.source_type === 'pdf' && typeof meta.total_pages === 'number') {
    parts.push(`${meta.total_pages} page${meta.total_pages === 1 ? '' : 's'}`)
  }
  if (doc.source_type === 'image' && typeof meta.width === 'number' && typeof meta.height === 'number') {
    parts.push(`${meta.width}×${meta.height}`)
  }

  return parts.join(' · ')
}

function DocumentCard({ doc }: { doc: NormalizedDocument }) {
  const [isOpen, setIsOpen] = useState(false)
  const Icon = SOURCE_ICON[doc.source_type] ?? FileText
  const hasDetail = doc.content.length > 0

  return (
    <div className="processed-doc">
      <button
        type="button"
        className="processed-doc__header"
        onClick={() => setIsOpen((open) => !open)}
        aria-expanded={isOpen}
        disabled={!hasDetail}
      >
        <span className={`processed-doc__icon processed-doc__icon--${doc.source_type}`}>
          <Icon size={15} strokeWidth={2} />
        </span>
        <span className="processed-doc__text">
          <span className="processed-doc__filename">{doc.filename}</span>
          <span className="processed-doc__summary">{summaryLine(doc)}</span>
        </span>
        {hasDetail && (
          <ChevronDown size={15} className={`processed-doc__chevron ${isOpen ? 'processed-doc__chevron--open' : ''}`} />
        )}
      </button>
      {doc.warnings.length > 0 && (
        <p className="processed-doc__warning">{doc.warnings.join('; ')}</p>
      )}
      {isOpen && hasDetail && (
        <pre className="processed-doc__preview">{doc.content.slice(0, 1200)}</pre>
      )}
    </div>
  )
}

/** Redesigned "processed documents" section -- expandable per-file cards. */
export function ProcessedDocuments({ documents }: ProcessedDocumentsProps) {
  if (documents.length === 0) return null

  return (
    <div className="processed-docs">
      <p className="processed-docs__label">
        Source material · {documents.length} file{documents.length === 1 ? '' : 's'}
      </p>
      <div className="processed-docs__list">
        {documents.map((doc) => (
          <DocumentCard key={doc.id} doc={doc} />
        ))}
      </div>
    </div>
  )
}
