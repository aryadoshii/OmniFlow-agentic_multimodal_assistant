"""Deterministic structural output validation (Phase 4.8).

Performs cheap, regex/structure-based checks against a synthesized answer --
never semantic correctness verification (that would require another LLM
call, which is explicitly out of scope here). Used by the graph's
validate_output node to decide whether a single bounded synthesis
correction pass is warranted.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

_BULLET_LINE_PATTERN = re.compile(r"^[ \t]*[-*•][ \t]+\S", re.MULTILINE)
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+(?=\s|$)")
_ONE_LINE_PATTERN = re.compile(r"\b(?:one|1)[\s-]line\b", re.IGNORECASE)
_REQUIRED_SECTION_PATTERN = re.compile(
    r"(?:section|include)[^\"'\n]{0,40}[\"']([^\"']{2,60})[\"']", re.IGNORECASE
)

# Shared qualifier vocabulary for count-style constraints ("exactly 3
# bullets", "5 sentences", "at least 2 bullets", "at most 3 sentences"). A
# bare count with no qualifier word (e.g. the assignment's literal "3
# bullets") is treated as "exactly" -- the most literal reading.
_QUALIFIER_GROUP = r"(exactly|at least|at most|no more than|maximum of|up to)?"
_BULLET_COUNT_PATTERN = re.compile(
    rf"{_QUALIFIER_GROUP}\s*(\d+)\s+bullet(?:s)?(?:\s+points?)?", re.IGNORECASE
)
_SENTENCE_COUNT_PATTERN = re.compile(rf"{_QUALIFIER_GROUP}\s*(\d+)\s+sentences?", re.IGNORECASE)

_AT_LEAST_QUALIFIERS = {"at least"}
_AT_MOST_QUALIFIERS = {"at most", "no more than", "maximum of", "up to"}


class ValidationResult(BaseModel):
    """Outcome of deterministic structural validation of a synthesized answer."""

    is_valid: bool = Field(..., description="True if no structural violation was found.")
    violations: list[str] = Field(
        default_factory=list,
        description="Concise, human-readable structural violations found, if any.",
    )


def _count_sentences(text: str) -> int:
    """Approximates sentence count by splitting on '.', '!', '?' runs.

    Not full sentence-boundary detection (abbreviations like "Dr." or
    decimals like "3.14" can mis-split) -- a deliberately cheap, bounded
    heuristic appropriate for a structural check, not semantic analysis.
    """
    candidates = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text)]
    return len([s for s in candidates if s])


def _check_count_constraint(
    constraint: str,
    pattern: re.Pattern[str],
    actual: int,
    unit: str,
    violations: list[str],
) -> None:
    """Shared logic for qualifier-aware count constraints (bullets, sentences)."""
    match = pattern.search(constraint)
    if not match:
        return

    qualifier = (match.group(1) or "exactly").lower()
    expected = int(match.group(2))

    if qualifier in _AT_LEAST_QUALIFIERS:
        ok, relation = actual >= expected, f"at least {expected}"
    elif qualifier in _AT_MOST_QUALIFIERS:
        ok, relation = actual <= expected, f"at most {expected}"
    else:
        ok, relation = actual == expected, f"exactly {expected}"

    if not ok:
        violations.append(
            f"Constraint '{constraint}' requires {relation} {unit}(s); found {actual}."
        )


def validate_structure(answer: str | None, constraints: list[str]) -> ValidationResult:
    """Runs deterministic structural checks against a synthesized answer.

    Checks performed (all purely structural -- never semantic correctness):

    - non_empty: the answer must contain non-whitespace text.
    - text_only: the answer must not be a raw JSON object/array dump.
    - bullet count: a constraint mentioning "N bullets" (optionally
      qualified with "exactly"/"at least"/"at most"/etc., defaulting to
      "exactly" for a bare count) is checked against the number of
      bullet-prefixed lines ("-", "*", "•") in the answer.
    - sentence count: same qualifier handling, for "N sentences" against an
      approximate sentence count (see _count_sentences).
    - one-line output: a constraint mentioning "one-line"/"1-line" requires
      the answer to contain exactly one non-blank line.
    - required sections: if a constraint names a quoted section (e.g.
      "include a 'Risks' section"), that phrase must literally appear in
      the answer (case-insensitive substring match).

    Args:
        answer: The synthesized final answer, or None if synthesis never
                produced one.
        constraints: Verbatim user constraints (from IntentResult.constraints).

    Returns:
        A ValidationResult. Absence of a violation is the only thing this
        function claims -- it never asserts the answer is factually or
        semantically correct.
    """
    if answer is None or not answer.strip():
        return ValidationResult(is_valid=False, violations=["Output is empty."])

    violations: list[str] = []
    stripped = answer.strip()

    looks_structured = (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    )
    if looks_structured:
        try:
            json.loads(stripped)
        except ValueError:
            pass
        else:
            violations.append("Output must be plain text, not a raw JSON/structured dump.")

    bullet_actual = len(_BULLET_LINE_PATTERN.findall(answer))
    sentence_actual = _count_sentences(answer)
    non_blank_lines = [line for line in answer.splitlines() if line.strip()]

    for constraint in constraints:
        _check_count_constraint(constraint, _BULLET_COUNT_PATTERN, bullet_actual, "bullet", violations)
        _check_count_constraint(
            constraint, _SENTENCE_COUNT_PATTERN, sentence_actual, "sentence", violations
        )

        if _ONE_LINE_PATTERN.search(constraint) and len(non_blank_lines) > 1:
            violations.append(
                f"Constraint '{constraint}' requires a single-line output; "
                f"found {len(non_blank_lines)} lines."
            )

        section_match = _REQUIRED_SECTION_PATTERN.search(constraint)
        if section_match:
            required_section = section_match.group(1)
            if required_section.lower() not in answer.lower():
                violations.append(
                    f"Constraint '{constraint}' requires a '{required_section}' section, "
                    "which was not found in the output."
                )

    return ValidationResult(is_valid=not violations, violations=violations)
