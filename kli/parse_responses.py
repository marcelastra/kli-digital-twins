"""Write model answers back into the response-JSON schema, verifiably.

ORIGINAL WORK (Kravis Leadership Institute). See ATTRIBUTION.md.

-------------------------------------------------------------------------------
TWO FIXES OVER THE UPSTREAM POSTPROCESSOR
-------------------------------------------------------------------------------

**1. Answers are matched by QuestionID, not by position.**

The vendored ``text_simulation/postprocess_responses.py`` re-walks the document
with its own counter and looks up ``response["Q{count}"]``. If that walk
diverges from the one that built the prompt, every later answer is attached to
the wrong question, silently. Here, both sides share
:func:`kli.survey_tree.walk_questions`, and every answer's echoed
``QuestionID`` is checked against the question actually asked. A mismatch
raises :class:`kli.survey_tree.RoundtripError`.

**2. A failed question is never left holding the human's answer.**

Upstream writes the output file even when parsing or validation fails, and a
failed question keeps its original ``Answers`` value -- the real respondent's
answer, in a file named as model output, with nothing marking it. Scored
against held-out data that inflates accuracy.

Here every question carries ``LLM_Imputed`` (bool) and ``LLM_Status``, the
human answer is preserved separately under ``Original_Answers``, and the
default on failure is to blank the answer so it reads as missing downstream.
``on_failure="keep_original"`` restores the upstream behavior, but the flags
still mark it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

from kli.survey_tree import QuestionRef, RoundtripError, verify_roundtrip

__all__ = [
    "ParseReport",
    "QuestionOutcome",
    "apply_responses",
    "validate_answers",
    "ValidationFailure",
]

OnFailure = Literal["blank", "keep_original"]


class ValidationFailure(ValueError):
    """Raised when a model answer does not fit its question's shape."""


@dataclass
class QuestionOutcome:
    """What happened to one question.

    Attributes:
        label: The ``Qn`` label.
        question_id: The Qualtrics QuestionID.
        imputed: True if the model's answer was accepted and written.
        status: ``"ok"``, ``"missing"`` (model omitted it), or
            ``"invalid: <reason>"``.
    """

    label: str
    question_id: str
    imputed: bool
    status: str


@dataclass
class ParseReport:
    """Summary of applying one model response to one participant.

    Attributes:
        participant_id: The participant.
        outcomes: Per-question results, in canonical order.
        roundtrip_ok: False if any QuestionID echo failed.
    """

    participant_id: str
    outcomes: list[QuestionOutcome] = field(default_factory=list)
    roundtrip_ok: bool = True

    @property
    def n_total(self) -> int:
        """Number of answerable questions."""
        return len(self.outcomes)

    @property
    def n_imputed(self) -> int:
        """Number of questions the model successfully answered."""
        return sum(1 for o in self.outcomes if o.imputed)

    @property
    def n_failed(self) -> int:
        """Number of questions the model did not successfully answer."""
        return self.n_total - self.n_imputed

    @property
    def complete(self) -> bool:
        """True if every question was imputed and the round trip held."""
        return self.roundtrip_ok and self.n_failed == 0

    def summary(self) -> str:
        """One-line human-readable summary."""
        state = "complete" if self.complete else "INCOMPLETE"
        return (
            f"pid {self.participant_id}: {self.n_imputed}/{self.n_total} imputed "
            f"({state})"
        )

    def failures(self) -> list[QuestionOutcome]:
        """The questions that were not imputed."""
        return [o for o in self.outcomes if not o.imputed]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value.strip().replace("%", "").replace(",", ""))
        except ValueError:
            return False
        return True
    return False


def validate_answers(ref: QuestionRef, answers: Any) -> dict[str, Any]:
    """Check and normalize a model answer for one question.

    Length-checks list answers against the question's own item count, which is
    what catches a model that silently drops matrix rows or slider statements.

    Args:
        ref: The question asked.
        answers: The ``Answers`` object the model returned.

    Returns:
        The normalized ``Answers`` dict to store.

    Raises:
        ValidationFailure: If the answer cannot be used.
    """
    if not isinstance(answers, dict):
        raise ValidationFailure(
            f"expected an Answers object, got {type(answers).__name__}"
        )

    question = ref.question
    qtype = ref.question_type
    selector = str(question.get("Settings", {}).get("Selector", "") or "").upper()

    if qtype == "MC":
        options = question.get("Options", []) or []
        n_options = len(options)
        multi = selector in {"MAVR", "MAHR", "MACOL"}
        positions = answers.get("SelectedByPosition")
        texts = answers.get("SelectedText")

        if positions is None and texts is None:
            raise ValidationFailure("no SelectedByPosition or SelectedText")

        pos_list = _as_list(positions)
        if not pos_list:
            raise ValidationFailure("SelectedByPosition is empty")
        if not all(_is_number(p) for p in pos_list):
            raise ValidationFailure(f"non-numeric SelectedByPosition: {pos_list!r}")

        ints = [int(float(str(p).strip())) for p in pos_list]
        if n_options and any(i < 1 or i > n_options for i in ints):
            raise ValidationFailure(
                f"SelectedByPosition {ints} outside 1..{n_options}"
            )
        if not multi and len(ints) != 1:
            raise ValidationFailure(
                f"single-choice question got {len(ints)} selections"
            )

        # Re-derive label text from the instrument rather than trusting the
        # model's transcription of it.
        labels = [options[i - 1] if n_options else None for i in ints]
        if multi:
            return {"SelectedByPosition": ints, "SelectedText": labels}
        return {"SelectedByPosition": ints[0], "SelectedText": labels[0]}

    if qtype == "Matrix":
        rows = question.get("Rows", []) or []
        columns = question.get("Columns", []) or []
        expected, n_cols = len(rows), len(columns)
        positions = _as_list(answers.get("SelectedByPosition"))

        if not positions:
            raise ValidationFailure("no SelectedByPosition")
        if expected and len(positions) != expected:
            raise ValidationFailure(
                f"expected {expected} row answers, got {len(positions)}"
            )
        if not all(_is_number(p) for p in positions):
            raise ValidationFailure(f"non-numeric row answers: {positions!r}")

        ints = [int(float(str(p).strip())) for p in positions]
        if n_cols and any(i < 1 or i > n_cols for i in ints):
            raise ValidationFailure(f"row answers {ints} outside 1..{n_cols}")

        labels = [columns[i - 1] if n_cols else None for i in ints]
        return {"SelectedByPosition": ints, "SelectedText": labels}

    if qtype == "Slider":
        statements = question.get("Statements", []) or []
        expected = len(statements)
        values = _as_list(answers.get("Values"))

        if not values:
            raise ValidationFailure("no Values")
        if expected and len(values) != expected:
            raise ValidationFailure(
                f"expected {expected} slider values, got {len(values)}"
            )
        if not all(_is_number(v) for v in values):
            raise ValidationFailure(f"non-numeric Values: {values!r}")

        rng = question.get("Range", {}) or {}
        lo, hi = rng.get("Min"), rng.get("Max")
        numbers = [float(str(v).strip()) for v in values]
        if lo is not None and hi is not None:
            try:
                lo_f, hi_f = float(lo), float(hi)
            except (TypeError, ValueError):
                lo_f = hi_f = None  # type: ignore[assignment]
            if lo_f is not None and any(n < lo_f or n > hi_f for n in numbers):
                raise ValidationFailure(f"Values {numbers} outside {lo}..{hi}")

        return {"Values": [str(v) for v in values]}

    if qtype == "CS":
        rows = question.get("Rows", []) or []
        expected = len(rows)
        values = _as_list(answers.get("Values"))

        if not values:
            raise ValidationFailure("no Values")
        if expected and len(values) != expected:
            raise ValidationFailure(
                f"expected {expected} values, got {len(values)}"
            )
        if not all(_is_number(v) for v in values):
            raise ValidationFailure(f"non-numeric Values: {values!r}")

        numbers = [float(str(v).strip()) for v in values]
        target = question.get("Settings", {}).get("ChoiceTotal")
        if target is not None:
            try:
                target_f = float(target)
            except (TypeError, ValueError):
                target_f = None
            # Tolerance of 1 absorbs integer rounding; a model that returns
            # 33/33/33 against a 100 total is accepted, one that returns
            # 50/50/50 is not.
            if target_f is not None and abs(sum(numbers) - target_f) > 1.0:
                raise ValidationFailure(
                    f"constant-sum values total {sum(numbers):g}, expected "
                    f"{target_f:g}"
                )

        return {"Values": [str(v) for v in values]}

    if qtype == "TE":
        if selector == "FORM":
            rows = question.get("Rows", []) or []
            expected = len(rows)
            texts = _as_list(answers.get("Text"))
            if not texts:
                raise ValidationFailure("no Text")
            if expected and len(texts) != expected:
                raise ValidationFailure(
                    f"expected {expected} fields, got {len(texts)}"
                )
            return {"Text": ["" if t is None else str(t) for t in texts]}

        text = answers.get("Text")
        if text is None:
            raise ValidationFailure("no Text")
        if isinstance(text, (list, tuple)):
            text = " ".join(str(t) for t in text if t is not None)
        text = str(text).strip()
        if not text:
            raise ValidationFailure("Text is empty")
        if selector == "ESTB" and not _is_number(text):
            raise ValidationFailure(f"numeric entry got non-numeric {text!r}")
        return {"Text": text}

    raise ValidationFailure(f"no validator for QuestionType {qtype!r}")


def apply_responses(
    refs: Sequence[QuestionRef],
    response: dict[str, Any],
    *,
    participant_id: str,
    on_failure: OnFailure = "blank",
    strict_roundtrip: bool = True,
) -> ParseReport:
    """Write a model response into the question dicts, in place.

    Each answerable question gains:

    * ``Original_Answers`` -- the human answer, preserved.
    * ``Answers`` -- the model's answer, or blanked on failure.
    * ``LLM_Imputed`` -- bool.
    * ``LLM_Status`` -- ``"ok"`` / ``"missing"`` / ``"invalid: ..."``.
    * ``LLM_Reasoning`` -- when the model supplied one.

    Args:
        refs: Questions for this response, from the shared traversal.
        response: The model's parsed JSON object, keyed by ``Qn``.
        participant_id: For the report.
        on_failure: ``"blank"`` (default) clears ``Answers`` so the question
            reads as missing downstream. ``"keep_original"`` leaves the human
            answer in place -- upstream's behavior; only use it if you
            understand that it puts real answers in a model-output file.
        strict_roundtrip: Require the model to echo each ``QuestionID``.

    Returns:
        A :class:`ParseReport`.

    Raises:
        RoundtripError: If an echoed QuestionID does not match. Not retryable:
            it means prompt and parse disagree about survey structure.
    """
    report = ParseReport(participant_id=participant_id)

    for ref in refs:
        if ref.is_descriptive:
            continue

        question = ref.question
        if "Original_Answers" not in question:
            question["Original_Answers"] = question.get("Answers")

        entry = response.get(ref.label)

        if not isinstance(entry, dict):
            status = "missing"
            if entry is not None:
                status = f"invalid: entry is {type(entry).__name__}, not an object"
            _mark_failed(question, on_failure, status)
            report.outcomes.append(
                QuestionOutcome(ref.label, ref.question_id, False, status)
            )
            continue

        try:
            verify_roundtrip(ref, entry, strict=strict_roundtrip)
        except RoundtripError:
            report.roundtrip_ok = False
            raise

        try:
            normalized = validate_answers(ref, entry.get("Answers"))
        except ValidationFailure as exc:
            status = f"invalid: {exc}"
            _mark_failed(question, on_failure, status)
            report.outcomes.append(
                QuestionOutcome(ref.label, ref.question_id, False, status)
            )
            continue

        question["Answers"] = normalized
        question["LLM_Imputed"] = True
        question["LLM_Status"] = "ok"
        reasoning = entry.get("Reasoning")
        if reasoning:
            question["LLM_Reasoning"] = str(reasoning)
        report.outcomes.append(
            QuestionOutcome(ref.label, ref.question_id, True, "ok")
        )

    return report


def _mark_failed(
    question: dict[str, Any], on_failure: OnFailure, status: str = "failed"
) -> None:
    """Flag a question as not imputed, and blank it unless told otherwise.

    The reason is written onto the question, not only into the run report, so
    that a stray imputed JSON file is self-describing: anyone opening it can see
    which answers are the model's and why the others are missing.
    """
    question["LLM_Imputed"] = False
    question["LLM_Status"] = status
    if on_failure == "blank":
        question["Answers"] = None


def aggregate_reports(reports: Iterable[ParseReport]) -> dict[str, Any]:
    """Roll per-participant reports into run-level counts.

    Args:
        reports: Per-participant reports.

    Returns:
        Totals plus the most common failure reasons, for the run manifest.
    """
    from collections import Counter

    reports = list(reports)
    reasons: Counter[str] = Counter()
    for report in reports:
        for outcome in report.failures():
            key = outcome.status.split(":")[0].strip()
            reasons[key] += 1

    total = sum(r.n_total for r in reports)
    imputed = sum(r.n_imputed for r in reports)
    return {
        "participants": len(reports),
        "participants_complete": sum(1 for r in reports if r.complete),
        "questions_total": total,
        "questions_imputed": imputed,
        "questions_failed": total - imputed,
        "imputation_rate": round(imputed / total, 4) if total else None,
        "failure_reasons": dict(reasons.most_common()),
    }
