import type { ToolExecutionTrace } from '../api/types'

/**
 * Presentation-only labeling for execution trace steps.
 *
 * This maps the backend's REAL step_name values (see omniflow/graph/nodes.py
 * -- "node:prepare_context", "node:understand_intent", "node:plan",
 * "tool:<name>", "node:synthesize", etc.) to human-readable labels and a
 * coarse stage grouping for grouping/coloring. It invents no data: every
 * label describes only what kind of step already ran, using only the
 * step_name/tool_name/status/details the backend actually returned.
 */

export type TraceStageGroup = 'ingest' | 'understanding' | 'planning' | 'tool' | 'synthesis' | 'other'

const STEP_LABELS: Record<string, string> = {
  'node:prepare_context': 'Preparing document context',
  'node:understand_intent': 'Understanding your request',
  'node:check_clarity': 'Checking whether clarification is needed',
  'node:clarification': 'Requesting clarification',
  'node:plan': 'Planning next step',
  'node:execute_tool': 'Evaluating whether a tool is needed',
  'node:observe_result': 'Reviewing tool result',
  'node:route_next': 'Deciding whether more work is needed',
  'node:synthesize': 'Synthesizing the answer',
  'node:validate_output': 'Validating output structure',
}

const STEP_GROUPS: Record<string, TraceStageGroup> = {
  'node:prepare_context': 'ingest',
  'node:understand_intent': 'understanding',
  'node:check_clarity': 'understanding',
  'node:clarification': 'understanding',
  'node:plan': 'planning',
  'node:execute_tool': 'planning',
  'node:observe_result': 'planning',
  'node:route_next': 'planning',
  'node:synthesize': 'synthesis',
  'node:validate_output': 'synthesis',
}

export interface TraceStepDisplay {
  label: string
  group: TraceStageGroup
}

export function describeTraceStep(step: ToolExecutionTrace): TraceStepDisplay {
  if (step.tool_name) {
    return { label: `Running tool: ${step.tool_name}`, group: 'tool' }
  }
  const label = STEP_LABELS[step.step_name] ?? step.step_name
  const group = STEP_GROUPS[step.step_name] ?? 'other'
  return { label, group }
}
