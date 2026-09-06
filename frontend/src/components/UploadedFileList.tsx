import { FileAudio, FileImage, FileText, X } from 'lucide-react'
import { formatBytes, guessFileKind } from '../lib/files'
import type { FileKind, StagedFile } from '../lib/files'
import './UploadedFileList.css'

interface UploadedFileListProps {
  files: StagedFile[]
  onRemove: (id: string) => void
  disabled: boolean
}

const KIND_ICON: Record<FileKind, typeof FileText> = {
  text: FileText,
  pdf: FileText,
  image: FileImage,
  audio: FileAudio,
  unknown: FileText,
}

const KIND_LABEL: Record<FileKind, string> = {
  text: 'TXT',
  pdf: 'PDF',
  image: 'IMG',
  audio: 'AUDIO',
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
        const Icon = KIND_ICON[kind]
        const classNames = ['uploaded-file-list__chip', issue && 'uploaded-file-list__chip--issue']
          .filter(Boolean)
          .join(' ')

        return (
          <li key={id} className={classNames}>
            <span className={`uploaded-file-list__kind uploaded-file-list__kind--${kind}`} aria-hidden="true">
              <Icon size={15} strokeWidth={2} />
            </span>
            <span className="uploaded-file-list__info">
              <span className="uploaded-file-list__name" title={file.name}>
                {file.name}
              </span>
              <span className="uploaded-file-list__meta">
                <span className="uploaded-file-list__badge">{KIND_LABEL[kind]}</span>
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
              <X size={14} />
            </button>
          </li>
        )
      })}
    </ul>
  )
}
