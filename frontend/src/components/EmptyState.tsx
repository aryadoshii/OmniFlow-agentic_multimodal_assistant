import { FileText, Headphones, Image, Link2 } from 'lucide-react'
import './EmptyState.css'

const CAPABILITIES = [
  { icon: FileText, title: 'Documents', description: 'Summarize or extract information from PDFs and text files.' },
  { icon: Image, title: 'Images', description: 'Understand screenshots, diagrams, and photos.' },
  { icon: Headphones, title: 'Audio', description: 'Transcribe and summarize spoken recordings.' },
  { icon: Link2, title: 'Cross-source reasoning', description: 'Compare and connect information across multiple files.' },
]

/** Shown before the first query -- explains what OmniFlow can actually do. */
export function EmptyState() {
  return (
    <div className="empty-state">
      <h3 className="empty-state__title">What can OmniFlow understand?</h3>
      <div className="empty-state__grid">
        {CAPABILITIES.map(({ icon: Icon, title, description }) => (
          <div key={title} className="empty-state__item">
            <Icon size={17} strokeWidth={1.75} className="empty-state__item-icon" aria-hidden="true" />
            <p className="empty-state__item-title">{title}</p>
            <p className="empty-state__item-description">{description}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
