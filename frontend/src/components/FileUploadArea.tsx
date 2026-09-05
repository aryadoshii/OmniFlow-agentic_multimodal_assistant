import { useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, KeyboardEvent } from 'react'
import './FileUploadArea.css'

// UX hint only (the `accept` attribute filters the OS file picker and is
// trivially bypassable, e.g. via drag-and-drop) -- the backend's
// IngestionService/processors remain the sole authority on what is
// actually supported; this is not a re-implementation of that validation.
// Kept in sync with the formats documented in README.md.
const ACCEPTED_EXTENSIONS = ['.txt', '.pdf', '.jpg', '.jpeg', '.png', '.wav', '.mp3', '.m4a']

interface FileUploadAreaProps {
  onFilesSelected: (files: File[]) => void
  disabled: boolean
}

/** Drag-and-drop + click-to-browse picker for one or more files. */
export function FileUploadArea({ onFilesSelected, disabled }: FileUploadAreaProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)

  function handleFiles(fileList: FileList | null) {
    if (!fileList || fileList.length === 0) return
    onFilesSelected(Array.from(fileList))
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setIsDragging(false)
    if (disabled) return
    handleFiles(event.dataTransfer.files)
  }

  function handleActivate() {
    if (!disabled) inputRef.current?.click()
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      handleActivate()
    }
  }

  const classNames = [
    'file-upload-area',
    isDragging && 'file-upload-area--dragging',
    disabled && 'file-upload-area--disabled',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <div
      className={classNames}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      onClick={handleActivate}
      onKeyDown={handleKeyDown}
      onDragOver={(event) => {
        event.preventDefault()
        if (!disabled) setIsDragging(true)
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPTED_EXTENSIONS.join(',')}
        className="file-upload-area__input"
        disabled={disabled}
        onChange={(event: ChangeEvent<HTMLInputElement>) => {
          handleFiles(event.target.files)
          // Reset so selecting the same file again still fires onChange.
          event.target.value = ''
        }}
      />
      <p className="file-upload-area__text">Drag files here, or click to browse</p>
      <p className="file-upload-area__hint">PDF · image (JPG/PNG) · audio (WAV/MP3/M4A) · text — multiple files supported</p>
    </div>
  )
}
