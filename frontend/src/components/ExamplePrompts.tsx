import { EXAMPLE_PROMPTS } from '../lib/promptExamples'
import './ExamplePrompts.css'

interface ExamplePromptsProps {
  onSelect: (query: string) => void
  disabled: boolean
}

export function ExamplePrompts({ onSelect, disabled }: ExamplePromptsProps) {
  return (
    <div className="example-prompts" aria-label="Example prompts">
      {EXAMPLE_PROMPTS.map((example) => (
        <button
          key={example.label}
          type="button"
          className="example-prompts__chip"
          onClick={() => onSelect(example.query)}
          disabled={disabled}
        >
          {example.label}
        </button>
      ))}
    </div>
  )
}
