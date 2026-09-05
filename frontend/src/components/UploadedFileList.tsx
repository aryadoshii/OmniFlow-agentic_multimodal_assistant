import { formatBytes, guessFileKind } from '../lib/files'
import type { StagedFile } from '../lib/files'
import './UploadedFileList.css'

interface UploadedFileListProps {
  files: StagedFile[]
  onRemove: (id: string) => void
  disabled: boolean
}

const KIND_BADGE: Record<string, string> = {
  text: 'TXT',
  pdf: 'PDF',
  image: 'IMG',
  audio: 'AUD',
  unknown: '?',
}

/** Removable chip/card per staged file, queued for the next submission. */
export function UploadedFileList({ files, onRemove, disabled }: UploadedFileListProps) {
  if (files.length === 0) {
    return null
  }

  return (
    <ul className="uploaded-file-list" aria-label="Attached files">
      {files.map(({ id, file, issue }) => {
        const kind = guessFileKind(file.name)
        const classNames = ['uploaded-file-list__chip', issue && 'uploaded-file-list__chip--issue']
          .filter(Boolean)
          .join(' ')

        return (
          <li key={id} className={classNames}>
            <span className={`uploaded-file-list__kind uploaded-file-list__kind--${kind}`} aria-hidden="true">
              {KIND_BADGE[kind]}
            </span>
            <span className="uploaded-file-list__info">
              <span className="uploaded-file-list__name" title={file.name}>
                {file.name}
              </span>
              <span className="uploaded-file-list__meta">
                {formatBytes(file.size)}
                {kind === 'unknown' && !issue && ' · format may not be supported'}
                {issue && ` · ${issue}`}
              </span>
            </span>
            <button
              type="button"
              className="uploaded-file-list__remove"
              onClick={() => onRemove(id)}
              disabled={disabled}
              aria-label={`Remove ${file.name}`}
            >
              ✕
            </button>
          </li>
        )
      })}
    </ul>
  )
}
