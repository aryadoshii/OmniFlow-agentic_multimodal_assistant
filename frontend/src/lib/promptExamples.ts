/** Rotating composer placeholders and clickable example-prompt chips. Pure
 * UI copy -- these only pre-fill the query text field; nothing here talks
 * to the backend or affects what gets sent. */

export const ROTATING_PLACEHOLDERS = [
  'Summarize these files in 5 bullet points.',
  'Compare the information across these sources.',
  'What are the key findings?',
  'Explain this like I’m a beginner.',
]

export interface ExamplePrompt {
  label: string
  query: string
}

export const EXAMPLE_PROMPTS: ExamplePrompt[] = [
  { label: 'Summarize', query: 'Summarize these files in 5 bullet points.' },
  { label: 'Compare sources', query: 'Compare the information across these sources.' },
  { label: 'Extract findings', query: 'Extract the key findings and any important numbers.' },
  { label: 'Explain simply', query: 'Explain this like I’m a beginner.' },
]
