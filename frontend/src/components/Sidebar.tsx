import { useMemo, useState } from 'react'
import { Plus, Search, Sparkles, Trash2, X } from 'lucide-react'
import type { ConversationSummary } from '../api/historyTypes'
import { groupConversationsByRecency } from '../lib/history'
import './Sidebar.css'

interface SidebarProps {
  conversations: ConversationSummary[]
  activeConversationId: string | null
  isLoading: boolean
  onSelectConversation: (id: string) => void
  onNewConversation: () => void
  onDeleteConversation: (id: string) => void
  isOpen: boolean
  onClose: () => void
}

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
  } catch {
    return ''
  }
}

export function Sidebar({
  conversations,
  activeConversationId,
  isLoading,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  isOpen,
  onClose,
}: SidebarProps) {
  const [query, setQuery] = useState('')

  const filtered = useMemo(() => {
    const trimmed = query.trim().toLowerCase()
    if (!trimmed) return conversations
    return conversations.filter((conversation) => conversation.title.toLowerCase().includes(trimmed))
  }, [conversations, query])

  const groups = useMemo(() => groupConversationsByRecency(filtered), [filtered])

  return (
    <>
      {isOpen && <div className="sidebar__scrim" onClick={onClose} aria-hidden="true" />}
      <aside className={`sidebar ${isOpen ? 'sidebar--open' : ''}`} aria-label="Conversation history">
        <div className="sidebar__top">
          <div className="sidebar__brand">
            <span className="sidebar__brand-mark" aria-hidden="true">
              <Sparkles size={18} strokeWidth={2.25} />
            </span>
            <div>
              <p className="sidebar__brand-title">OmniFlow</p>
              <p className="sidebar__brand-subtitle">Multimodal AI workspace</p>
            </div>
          </div>
          <button type="button" className="sidebar__close" onClick={onClose} aria-label="Close sidebar">
            <X size={18} />
          </button>
        </div>

        <button type="button" className="sidebar__new-btn" onClick={onNewConversation}>
          <Plus size={16} strokeWidth={2.5} />
          New conversation
        </button>

        <div className="sidebar__search">
          <Search size={14} className="sidebar__search-icon" aria-hidden="true" />
          <input
            type="search"
            placeholder="Search history…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            aria-label="Search conversation history"
          />
        </div>

        <div className="sidebar__history thin-scroll">
          <p className="sidebar__section-label">History</p>

          {isLoading && <p className="sidebar__empty-note">Loading…</p>}

          {!isLoading && groups.length === 0 && (
            <p className="sidebar__empty-note">
              {query ? 'No conversations match your search.' : 'Your conversations will appear here.'}
            </p>
          )}

          {groups.map((group) => (
            <div key={group.label} className="sidebar__group">
              <p className="sidebar__group-label">{group.label}</p>
              <ul className="sidebar__list">
                {group.items.map((conversation) => (
                  <li key={conversation.id}>
                    <button
                      type="button"
                      className={`sidebar__item ${
                        conversation.id === activeConversationId ? 'sidebar__item--active' : ''
                      }`}
                      onClick={() => onSelectConversation(conversation.id)}
                    >
                      <span className="sidebar__item-text">
                        <span className="sidebar__item-title">{conversation.title}</span>
                        <span className="sidebar__item-meta">
                          {formatTimestamp(conversation.updated_at)}
                          {conversation.attachment_count > 0 &&
                            ` · ${conversation.attachment_count} file${conversation.attachment_count === 1 ? '' : 's'}`}
                        </span>
                      </span>
                      <span
                        role="button"
                        tabIndex={0}
                        className="sidebar__item-delete"
                        aria-label={`Delete conversation "${conversation.title}"`}
                        onClick={(event) => {
                          event.stopPropagation()
                          onDeleteConversation(conversation.id)
                        }}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault()
                            event.stopPropagation()
                            onDeleteConversation(conversation.id)
                          }
                        }}
                      >
                        <Trash2 size={14} />
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </aside>
    </>
  )
}
