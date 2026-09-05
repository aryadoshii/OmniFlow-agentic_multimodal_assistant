/**
 * Composes a follow-up query after the backend asks for clarification.
 *
 * IMPORTANT backend limitation (see the Phase 5.2 report): OmniFlow has no
 * server-side conversation/session store -- `session_id` is accepted and
 * echoed back by POST /query but never used to look up prior state (grep
 * the backend: it only appears in logging and pass-through response
 * fields). Every /query call builds a brand-new AgentState from scratch.
 *
 * So "continuing" after a clarification question cannot be a real stateful
 * continuation. Instead, this folds the original request, the question
 * asked, and the user's answer into ONE new self-contained query string,
 * and the caller resubmits it (with the same staged files) to the SAME
 * /query endpoint. This is pure text composition -- no decision-making,
 * planning, or tool selection happens here; all of that still happens
 * exactly once, server-side, on the resulting request.
 */
export function buildClarificationFollowUp(
  originalQuery: string,
  clarificationQuestion: string | null,
  answer: string,
): string {
  const question = clarificationQuestion ?? 'the clarification request above'
  return [
    `Original request: ${originalQuery}`,
    `Clarification requested: ${question}`,
    `User's answer: ${answer}`,
  ].join('\n\n')
}
