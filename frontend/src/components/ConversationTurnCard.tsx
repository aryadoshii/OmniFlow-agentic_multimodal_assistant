import type { ConversationTurn } from '../lib/conversation'
import { ResponseArea } from './ResponseArea'
import './ConversationTurnCard.css'

interface ConversationTurnCardProps {
  turn: ConversationTurn
  /** The most recent turn is shown at full prominence; earlier ones are compact. */
  isLatest: boolean
}

export function ConversationTurnCard({ turn, isLatest }: ConversationTurnCardProps) {
  const classNames = ['conversation-turn', !isLatest && 'conversation-turn--compact'].filter(Boolean).join(' ')

  return (
    <article className={classNames} aria-label={`${turn.isClarificationFollowUp ? 'Answer' : 'Question'}: ${turn.displayQuery}`}>
      <header className="conversation-turn__query">
        <span className="conversation-turn__query-label">
          {turn.isClarificationFollowUp ? 'You answered' : 'You asked'}
        </span>
        <p className="conversation-turn__query-text">{turn.displayQuery}</p>
        {turn.fileNames.length > 0 && (
          <p className="conversation-turn__query-files">
            Attached: {turn.fileNames.join(', ')}
          </p>
        )}
      </header>
      <ResponseArea status={turn.status} response={turn.response} error={turn.error} />
    </article>
  )
}
