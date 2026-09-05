import type { ApiError } from '../api/client'

/**
 * Friendlier headings per backend error code (see omniflow/exceptions.py).
 * These only categorize/label -- the backend's own `message` (already
 * written to be user-facing, see the exception classes' docstrings) is
 * always shown alongside, never replaced or guessed at.
 */
const ERROR_TITLES: Record<string, string> = {
  INVALID_INPUT: 'Invalid request',
  UNSUPPORTED_FILE_TYPE: 'Unsupported file type',
  UPLOAD_VALIDATION_ERROR: 'File upload rejected',
  PROCESSING_FAILURE: 'Could not process a file',
  OCR_PROCESSING_ERROR: 'Text extraction failed',
  TRANSCRIPTION_ERROR: 'Audio transcription failed',
  TRANSCRIPT_UNAVAILABLE: 'No transcript available',
  EMBEDDING_GENERATION_ERROR: 'Search indexing failed',
  RAG_RETRIEVAL_ERROR: 'Document search failed',
  TOOL_EXECUTION_ERROR: 'A tool failed while answering',
  TOOL_NOT_FOUND: 'Requested capability unavailable',
  EXTERNAL_PROVIDER_ERROR: 'The AI provider is unavailable',
  CONFIGURATION_ERROR: 'Backend is not configured',
  ORCHESTRATION_ERROR: 'Agent execution failed',
  VALIDATION_ERROR: 'Invalid request',
  TIMEOUT: 'Request timed out',
  NETWORK_ERROR: 'Could not reach the server',
  UNKNOWN_ERROR: 'Something went wrong',
  INTERNAL_SERVER_ERROR: 'Server error',
}

export function describeApiError(error: ApiError): { title: string; message: string } {
  return {
    title: ERROR_TITLES[error.code] ?? 'Request failed',
    message: error.message,
  }
}

/**
 * Correlates a file-related error back to one of the currently staged
 * files, using ONLY the filename the backend's own error details actually
 * report (details.filename, or details.audio_file for transcription
 * errors -- see WhisperService.transcribe()). Returns null rather than
 * guessing when the backend didn't identify a file (e.g. OCR failures,
 * which the backend does not currently attribute to a filename -- see the
 * Phase 5.2 report).
 */
export function findRejectedFilename(error: ApiError): string | null {
  const filename = error.details.filename ?? error.details.audio_file
  return typeof filename === 'string' ? filename : null
}
