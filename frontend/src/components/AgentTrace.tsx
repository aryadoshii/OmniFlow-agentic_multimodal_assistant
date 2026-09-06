import { useState } from 'react'
import { ChevronDown, CircleAlert, CircleCheck, Loader } from 'lucide-react'
import type { ExecutionTrace } from '../api/types'
import { describeTraceStep } from '../lib/traceDisplay'
import './AgentTrace.css'

interface AgentTraceProps {
  trace: ExecutionTrace
}

/** Collapsible vertical timeline of the agent's execution steps -- only
 * ever renders the backend's own step_name/status/duration/error_message
 * metadata (see traceDisplay.ts). No hidden chain-of-thought is shown or
 * available to show. */
export function AgentTrace({ trace }: AgentTraceProps) {
  const [isOpen, setIsOpen] = useState(false)
  const stepCount = trace.steps.length
  if (stepCount === 0) return null

  return (
    <div className="agent-trace">
      <button
        type="button"
        className="agent-trace__toggle"
        onClick={() => setIsOpen((open) => !open)}
        aria-expanded={isOpen}
      >
        <span className="agent-trace__toggle-text">
          <span className="agent-trace__toggle-title">Agent activity</span>
          <span className="agent-trace__toggle-meta">
            {stepCount} step{stepCount === 1 ? '' : 's'}
            {trace.total_duration_ms != null && ` · ${(trace.total_duration_ms / 1000).toFixed(1)}s`}
          </span>
        </span>
        <ChevronDown size={16} className={`agent-trace__chevron ${isOpen ? 'agent-trace__chevron--open' : ''}`} />
      </button>

      {isOpen && (
        <ol className="agent-trace__timeline">
          {trace.steps.map((step, index) => {
            const { label } = describeTraceStep(step)
            const isLast = index === trace.steps.length - 1
            return (
              <li key={index} className="agent-trace__item">
                <span className="agent-trace__rail">
                  <span className={`agent-trace__dot agent-trace__dot--${step.status}`}>
                    {step.status === 'success' && <CircleCheck size={13} strokeWidth={2.5} />}
                    {step.status === 'failed' && <CircleAlert size={13} strokeWidth={2.5} />}
                    {step.status === 'pending' && <Loader size={12} strokeWidth={2.5} />}
                  </span>
                  {!isLast && <span className="agent-trace__connector" />}
                </span>
                <span className="agent-trace__content">
                  <span className="agent-trace__label">{label}</span>
                  <span className="agent-trace__step-meta">
                    {step.duration_ms != null && `${step.duration_ms.toFixed(0)} ms`}
                    {step.error_message && (
                      <span className="agent-trace__error"> · {step.error_message}</span>
                    )}
                  </span>
                </span>
              </li>
            )
          })}
        </ol>
      )}
    </div>
  )
}
