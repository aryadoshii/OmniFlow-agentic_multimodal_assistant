"""Tests for deterministic structural output validation (Phase 4.8).

No LLM/network is involved anywhere in this module -- validate_structure()
is a pure function over (answer, constraints).
"""

from omniflow.agents.validator import validate_structure


class TestNonEmpty:
    def test_none_answer_is_invalid(self) -> None:
        result = validate_structure(None, [])
        assert result.is_valid is False
        assert "empty" in result.violations[0].lower()

    def test_blank_answer_is_invalid(self) -> None:
        result = validate_structure("   \n  ", [])
        assert result.is_valid is False

    def test_non_empty_answer_with_no_constraints_is_valid(self) -> None:
        result = validate_structure("Here is a plain answer.", [])
        assert result.is_valid is True
        assert result.violations == []


class TestTextOnly:
    def test_raw_json_object_is_invalid(self) -> None:
        result = validate_structure('{"answer": "hello"}', [])
        assert result.is_valid is False
        assert any("json" in v.lower() for v in result.violations)

    def test_raw_json_array_is_invalid(self) -> None:
        result = validate_structure('["a", "b", "c"]', [])
        assert result.is_valid is False

    def test_text_that_merely_contains_braces_is_not_flagged(self) -> None:
        """Only a WHOLE-answer JSON dump is a violation -- ordinary prose
        that happens to mention braces must not be penalized."""
        result = validate_structure("The config uses a {key: value} style syntax.", [])
        assert result.is_valid is True

    def test_malformed_json_looking_text_is_not_flagged(self) -> None:
        """Starts/ends with braces but isn't valid JSON -- not a real dump."""
        result = validate_structure("{this is not json, just prose}", [])
        assert result.is_valid is True


class TestBulletCount:
    def test_exact_bullet_count_satisfied(self) -> None:
        answer = "- first point\n- second point\n- third point"
        result = validate_structure(answer, ["exactly 3 bullets"])
        assert result.is_valid is True

    def test_exact_bullet_count_violated_too_few(self) -> None:
        answer = "- first point\n- second point"
        result = validate_structure(answer, ["exactly 3 bullets"])
        assert result.is_valid is False
        assert "exactly 3 bullet" in result.violations[0]

    def test_exact_bullet_count_violated_too_many(self) -> None:
        answer = "- a\n- b\n- c\n- d"
        result = validate_structure(answer, ["exactly 3 bullets"])
        assert result.is_valid is False

    def test_asterisk_and_unicode_bullets_counted(self) -> None:
        answer = "* one\n• two"
        result = validate_structure(answer, ["exactly 2 bullets"])
        assert result.is_valid is True

    def test_constraint_without_bullet_requirement_ignores_bullets(self) -> None:
        answer = "- only one bullet here"
        result = validate_structure(answer, ["for a beginner audience"])
        assert result.is_valid is True

    def test_bare_count_with_no_qualifier_treated_as_exact(self) -> None:
        """The assignment's own literal example is '3 bullets', not
        'exactly 3 bullets' -- a bare count must still be enforced."""
        answer = "- a\n- b\n- c"
        result = validate_structure(answer, ["3 bullets"])
        assert result.is_valid is True

        result_violated = validate_structure("- a\n- b", ["3 bullets"])
        assert result_violated.is_valid is False

    def test_at_least_qualifier_allows_more(self) -> None:
        answer = "- a\n- b\n- c\n- d"
        result = validate_structure(answer, ["at least 3 bullets"])
        assert result.is_valid is True

    def test_at_least_qualifier_rejects_fewer(self) -> None:
        answer = "- a\n- b"
        result = validate_structure(answer, ["at least 3 bullets"])
        assert result.is_valid is False

    def test_at_most_qualifier_allows_fewer(self) -> None:
        answer = "- a"
        result = validate_structure(answer, ["at most 3 bullets"])
        assert result.is_valid is True

    def test_at_most_qualifier_rejects_more(self) -> None:
        answer = "- a\n- b\n- c\n- d"
        result = validate_structure(answer, ["no more than 3 bullets"])
        assert result.is_valid is False

    def test_bullet_points_phrasing_recognized(self) -> None:
        answer = "- a\n- b"
        result = validate_structure(answer, ["exactly 2 bullet points"])
        assert result.is_valid is True


class TestSentenceCount:
    def test_exact_sentence_count_satisfied(self) -> None:
        answer = "This is one. This is two. This is three."
        result = validate_structure(answer, ["exactly 3 sentences"])
        assert result.is_valid is True

    def test_exact_sentence_count_violated(self) -> None:
        answer = "This is one. This is two."
        result = validate_structure(answer, ["exactly 3 sentences"])
        assert result.is_valid is False
        assert "sentence" in result.violations[0].lower()

    def test_bare_sentence_count_treated_as_exact(self) -> None:
        answer = "One. Two. Three. Four. Five."
        result = validate_structure(answer, ["5 sentences"])
        assert result.is_valid is True

    def test_at_most_sentence_qualifier(self) -> None:
        answer = "One. Two."
        result = validate_structure(answer, ["at most 3 sentences"])
        assert result.is_valid is True


class TestOneLineConstraint:
    def test_one_line_answer_satisfies_constraint(self) -> None:
        result = validate_structure("A single concise line.", ["one-line summary"])
        assert result.is_valid is True

    def test_multi_line_answer_violates_one_line_constraint(self) -> None:
        answer = "First line.\nSecond line."
        result = validate_structure(answer, ["1-line summary"])
        assert result.is_valid is False
        assert "single-line" in result.violations[0]

    def test_blank_lines_do_not_count_toward_line_count(self) -> None:
        answer = "Only real content line.\n\n\n"
        result = validate_structure(answer, ["one-line summary"])
        assert result.is_valid is True


class TestRequiredSections:
    def test_required_section_present(self) -> None:
        answer = "Summary\nAll good.\n\nRisks\nNone found."
        result = validate_structure(answer, ["include a 'Risks' section"])
        assert result.is_valid is True

    def test_required_section_missing(self) -> None:
        answer = "Just a plain summary with no headers."
        result = validate_structure(answer, ["include a 'Risks' section"])
        assert result.is_valid is False
        assert "Risks" in result.violations[0]

    def test_required_section_case_insensitive_match(self) -> None:
        answer = "RISKS: none identified."
        result = validate_structure(answer, ["include a 'Risks' section"])
        assert result.is_valid is True


class TestMultipleConstraints:
    def test_multiple_violations_all_reported(self) -> None:
        answer = "- only one bullet"
        result = validate_structure(
            answer, ["exactly 3 bullets", "include a 'Conclusion' section"]
        )
        assert result.is_valid is False
        assert len(result.violations) == 2

    def test_one_satisfied_one_violated(self) -> None:
        answer = "- a\n- b\n- c"
        result = validate_structure(
            answer, ["exactly 3 bullets", "include a 'Conclusion' section"]
        )
        assert result.is_valid is False
        assert len(result.violations) == 1
