import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, FormEvent, KeyboardEvent } from 'react'
import { ArrowUp, Paperclip } from 'lucide-react'
import { ROTATING_PLACEHOLDERS } from '../lib/promptExamples'
import './QueryInput.css'

const ACCEPTED_EXTENSIONS = ['.txt', '.pdf', '.jpg', '.jpeg', '.png', '.wav', '.mp3', '.m4a']
const PLACEHOLDER_ROTATE_MS = 4200

interface QueryInputProps {
  value: string
  onValueChange: (value: string) => void
  onSubmit: (text: string) => void
  onAttachFiles: (files: File[]) => void
  disabled: boolean
  /** 'clarify' when the last response is awaiting a clarifying answer. */
  mode: 'ask' | 'clarify'
  clarificationQuestion?: string | null
}

/** The user's question/instruction text area plus a submit action. Fully
 * controlled: the parent owns `value` so an example-prompt click can set
 * it directly, with no extra effect-driven sync needed. */
export function QueryInput({
  value,
  onValueChange,
  onSubmit,
  onAttachFiles,
  disabled,
  mode,
  clarificationQuestion,
}: QueryInputProps) {
  const [placeholderIndex, setPlaceholderIndex] = useState(0)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const attachInputRef = useRef<HTMLInputElement>(null)

  const isClarifying = mode === 'clarify'

  // Focus the input when a clarification appears, so the user can answer
  // immediately without reaching for the mouse.
  useEffect(() => {
    if (isClarifying) {
      textareaRef.current?.focus()
    }
  }, [isClarifying])

  // Rotate the idle placeholder through a few example queries -- purely
  // cosmetic, paused once the user has typed anything.
  useEffect(() => {
    if (value || isClarifying) return
    const timer = setInterval(() => {
      setPlaceholderIndex((index) => (index + 1) % ROTATING_PLACEHOLDERS.length)
    }, PLACEHOLDER_ROTATE_MS)
    return () => clearInterval(timer)
  }, [value, isClarifying])

  function submit() {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSubmit(trimmed)
    onValueChange('')
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    submit()
  }

  // Enter submits; Shift+Enter inserts a newline -- standard chat-input
  // convention, and the only keyboard affordance beyond native
  // tab/click/space that a textarea needs. `isComposing` guards IME
  // composition (Japanese/Chinese/Korean, etc.): the Enter that confirms a
  // candidate there must never also submit the form.
  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      submit()
    }
  }

  function handleAttachChange(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files && event.target.files.length > 0) {
      onAttachFiles(Array.from(event.target.files))
    }
    event.target.value = ''
  }

  return (
    <form className="query-input" onSubmit={handleSubmit}>
      <label
        htmlFor="query-textarea"
        className={isClarifying ? 'query-input__label' : 'query-input__label query-input__label--visually-hidden'}
      >
        {isClarifying ? 'OmniFlow needs more information' : 'Ask OmniFlow'}
      </label>
      {isClarifying && clarificationQuestion && (
        <p className="query-input__clarification-echo">{clarificationQuestion}</p>
      )}
      <textarea
        ref={textareaRef}
        id="query-textarea"
        className="query-input__textarea"
        value={value}
        onChange={(event) => onValueChange(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={
          isClarifying ? 'Type your answer…' : `${ROTATING_PLACEHOLDERS[placeholderIndex]}`
        }
        rows={3}
        disabled={disabled}
      />
      <div className="query-input__toolbar">
        <div className="query-input__toolbar-left">
          <input
            ref={attachInputRef}
            type="file"
            multiple
            accept={ACCEPTED_EXTENSIONS.join(',')}
            className="query-input__hidden-input"
            disabled={disabled}
            onChange={handleAttachChange}
          />
          <button
            type="button"
            className="query-input__attach-btn"
            onClick={() => attachInputRef.current?.click()}
            disabled={disabled}
          >
            <Paperclip size={14} />
            Attach files
          </button>
        </div>
        <button
          type="submit"
          className="query-input__submit"
          disabled={disabled || !value.trim()}
          aria-label={disabled ? 'Working' : isClarifying ? 'Send answer' : 'Send'}
        >
          {disabled ? (
            <span className="query-input__spinner" aria-hidden="true" />
          ) : (
            <ArrowUp size={16} strokeWidth={2.5} />
          )}
        </button>
      </div>
      <p className="query-input__hint">
        <kbd>Enter</kbd> to send · <kbd>Shift</kbd> + <kbd>Enter</kbd> for a new line
      </p>
    </form>
  )
}
