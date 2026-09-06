import { useState } from 'react'
import { ExamplePrompts } from './ExamplePrompts'
import { FileUploadArea } from './FileUploadArea'
import { QueryInput } from './QueryInput'
import { UploadedFileList } from './UploadedFileList'
import type { StagedFile } from '../lib/files'
import './Workspace.css'

interface WorkspaceProps {
  files: StagedFile[]
  onFilesSelected: (files: File[]) => void
  onRemoveFile: (id: string) => void
  onSubmit: (text: string) => void
  disabled: boolean
  mode: 'ask' | 'clarify'
  clarificationQuestion?: string | null
}

/** The upload + query composer -- one seamless floating surface, the
 * visual focal point of the main screen. */
export function Workspace({
  files,
  onFilesSelected,
  onRemoveFile,
  onSubmit,
  disabled,
  mode,
  clarificationQuestion,
}: WorkspaceProps) {
  const [queryValue, setQueryValue] = useState('')

  return (
    <div className="composer-block">
      <section className="composer" aria-label="Query and file input">
        {files.length > 0 && <UploadedFileList files={files} onRemove={onRemoveFile} disabled={disabled} />}

        <FileUploadArea onFilesSelected={onFilesSelected} disabled={disabled} compact={files.length > 0} />

        <QueryInput
          value={queryValue}
          onValueChange={setQueryValue}
          onSubmit={onSubmit}
          onAttachFiles={onFilesSelected}
          disabled={disabled}
          mode={mode}
          clarificationQuestion={clarificationQuestion}
        />
      </section>

      {mode === 'ask' && <ExamplePrompts onSelect={setQueryValue} disabled={disabled} />}
    </div>
  )
}
