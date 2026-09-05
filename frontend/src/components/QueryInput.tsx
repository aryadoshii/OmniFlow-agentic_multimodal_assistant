import { useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import './QueryInput.css'

interface QueryInputProps {
  onSubmit: (text: string) => void
  disabled: boolean
  /** 'clarify' when the last response is awaiting a clarifying answer. */
  mode: 'ask' | 'clarify'
  clarificationQuestion?: string | null
}

/** The user's question/instruction text area plus a submit action. */
export function QueryInput({ onSubmit, disabled, mode, clarificationQuestion }: QueryInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Focus the input when a clarification appears, so the user can answer
  // immediately without reaching for the mouse.
  useEffect(() => {
    if (mode === 'clarify') {
      textareaRef.current?.focus()
    }
  }, [mode])

  function submit() {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSubmit(trimmed)
    setValue('')
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

  const isClarifying = mode === 'clarify'

  return (
    <form className="query-input" onSubmit={handleSubmit}>
      <label htmlFor="query-textarea" className="query-input__label">
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
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={
          isClarifying
            ? 'Type your answer…'
            : 'e.g. Summarize this PDF in 3 bullets, or ask a question about your files. (Enter to send, Shift+Enter for a new line)'
        }
        rows={4}
        disabled={disabled}
      />
      <div className="query-input__actions">
        <button type="submit" className="query-input__submit" disabled={disabled || !value.trim()}>
          {disabled ? 'Working…' : isClarifying ? 'Send answer' : 'Send'}
        </button>
      </div>
    </form>
  )
}
