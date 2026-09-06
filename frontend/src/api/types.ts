/**
 * Types mirroring OmniFlow's actual FastAPI/Pydantic response schemas.
 *
 * Kept in exact correspondence with the backend models -- do not add fields
 * here that the backend doesn't return, and update both sides together if
 * the backend contract changes:
 *   - backend/models/document.py    (NormalizedDocument)
 *   - backend/models/trace.py       (ToolExecutionTrace, ExecutionTrace)
 *   - backend/models/evidence.py    (EvidenceReference)
 *   - backend/agents/cross_source.py (CrossSourceAnalysis, SourceUnderstanding)
 *   - backend/models/response.py    (OmniFlowResponse)
 *   - backend/api/routes/ingest.py  (IngestionResponse)
 *   - backend/api/error_handlers.py (error envelope)
 */

export type SourceType = 'text' | 'image' | 'pdf' | 'audio'

export type ExtractionMethod =
  | 'direct_input'
  | 'native_text'
  | 'ocr'
  | 'mixed'
  | 'speech_to_text'

/** backend.models.document.NormalizedDocument */
export interface NormalizedDocument {
  id: string
  filename: string
  source_type: SourceType
  mime_type: string
  content: string
  extraction_method: ExtractionMethod
  confidence: number | null
  metadata: Record<string, unknown>
  detected_urls: string[]
  warnings: string[]
}

export type ToolExecutionStatus = 'pending' | 'success' | 'failed'

/** backend.models.trace.ToolExecutionTrace */
export interface ToolExecutionTrace {
  step_name: string
  tool_name: string | null
  status: ToolExecutionStatus
  duration_ms: number | null
  details: Record<string, string>
  error_message: string | null
}

/** backend.models.trace.ExecutionTrace */
export interface ExecutionTrace {
  session_id: string | null
  total_duration_ms: number | null
  steps: ToolExecutionTrace[]
}

/**
 * backend.models.state.WorkflowStatus values, as serialized by
 * OmniFlowResponse.status. The backend field is typed as a plain `str` (not
 * a strict enum) at the API boundary, so an unrecognized future value must
 * not crash the UI -- treat this as a hint, not an exhaustive union.
 */
export type WorkflowStatus = 'completed' | 'failed' | 'awaiting_clarification'

/** Which part of the pipeline a piece of evidence came from -- backend.models.evidence.EvidenceReference.source */
export type EvidenceSource = 'rag' | 'document' | 'youtube_transcript'

/** backend.models.evidence.EvidenceReference */
export interface EvidenceReference {
  source: EvidenceSource
  document_id: string | null
  filename: string | null
  source_type: string | null
  extraction_method: string | null
  page: number | null
  segment_start_seconds: number | null
  segment_end_seconds: number | null
  chunk_id: string | null
  score: number | null
  excerpt: string
}

/** backend.agents.cross_source.RelationshipType */
export type CrossSourceRelationship =
  | 'strong_overlap'
  | 'partial_overlap'
  | 'different_topics'
  | 'contradiction'
  | 'insufficient_evidence'

/** backend.agents.cross_source.SourceUnderstanding */
export interface SourceUnderstanding {
  document_id: string
  filename: string
  topic: string
  evidence: string
}

/** backend.agents.cross_source.CrossSourceAnalysis */
export interface CrossSourceAnalysis {
  relationship: CrossSourceRelationship
  sources: SourceUnderstanding[]
  shared_concepts: string[]
  differences: string[]
  explanation: string
}

/** backend.models.response.OmniFlowResponse (POST /query response body) */
export interface OmniFlowResponse {
  session_id: string | null
  status: WorkflowStatus | (string & {})
  answer: string | null
  clarification_needed: boolean
  clarification_prompt: string | null
  normalized_documents: NormalizedDocument[]
  execution_trace: ExecutionTrace | null
  warnings: string[]
  errors: string[]
  evidence: EvidenceReference[]
  cross_source_analysis: CrossSourceAnalysis | null
}

/** IngestionResponse (POST /ingest response body) */
export interface IngestionResponse {
  documents: NormalizedDocument[]
  total_count: number
  warnings: string[]
}

/** Error envelope shared by every OmniFlow error response (see error_handlers.py) */
export interface ApiErrorPayload {
  error: {
    code: string
    message: string
    details: Record<string, unknown>
  }
}
