/**
 * Client-side file staging helpers. These improve UX only (a stable id for
 * React keys/removal, a size label, a best-guess format badge) -- the
 * backend's IngestionService/file_validator.py remains the sole authority
 * on whether a file is actually accepted; nothing here re-implements or
 * duplicates that validation.
 */

// Mirrors the extensions file_validator.py's EXTENSION_TO_MIME accepts.
// Used only for an UP-FRONT, non-blocking "this looks unsupported" hint --
// the backend's own rejection (if any) is always what's authoritative and
// shown to the user.
const KNOWN_EXTENSIONS: Record<string, 'text' | 'pdf' | 'image' | 'audio'> = {
  '.txt': 'text',
  '.pdf': 'pdf',
  '.jpg': 'image',
  '.jpeg': 'image',
  '.png': 'image',
  '.wav': 'audio',
  '.mp3': 'audio',
  '.m4a': 'audio',
}

export type FileKind = 'text' | 'pdf' | 'image' | 'audio' | 'unknown'

export interface StagedFile {
  id: string
  file: File
  /** Set from a backend error correlated to this file's name (see errorDisplay.findRejectedFilename). Null until then. */
  issue: string | null
}

function extensionOf(filename: string): string {
  const dotIndex = filename.lastIndexOf('.')
  return dotIndex === -1 ? '' : filename.slice(dotIndex).toLowerCase()
}

export function guessFileKind(filename: string): FileKind {
  return KNOWN_EXTENSIONS[extensionOf(filename)] ?? 'unknown'
}

export function createStagedFile(file: File): StagedFile {
  const randomSuffix = Math.random().toString(36).slice(2, 8)
  return {
    id: `${file.name}-${file.size}-${file.lastModified}-${randomSuffix}`,
    file,
    issue: null,
  }
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
