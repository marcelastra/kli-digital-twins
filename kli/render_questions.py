"""Render survey questions into an answerable prompt with an explicit JSON skeleton.

ORIGINAL WORK (Kravis Leadership Institute). Not derived from either upstream
repository, though the output schema is deliberately kept compatible with the
``Answers`` shapes that ``processing_qualtrics_csv/`` writes and that
``evaluation/json2csv.py`` reads. See ATTRIBUTION.md.

-------------------------------------------------------------------------------
DESIGN NOTES
-------------------------------------------------------------------------------

**Question-type coverage.** Handles ``MC`` (single and multi-select),
``Matrix``, ``Slider``, ``TE`` (SL/ML/ESTB/FORM) and ``CS`` (constant sum),
plus ``DB`` as un-answered context. The vendored demo renderer covers five of
these and raises ``ValueError`` on ``CS``. That is not hypothetical: the
``story_beliefs`` instrument contains 2,008 constant-sum questions across 251
participants, so the vendored renderer cannot process it at all.

**Skeleton-filling instead of schema description.** Upstream describes the
response format in prose and gives per-type examples, then asks the model to
construct the JSON. We instead emit the *exact* JSON object we expect with
``null`` in every answer slot, and ask the model to return it with the nulls
replaced. This matters much more for a local 7B-class model than for GPT-4.1:
the model never has to infer structure, only fill values. Combined with
Ollama's ``format: "json"`` it makes malformed output rare.

**QuestionID echo.** Every entry carries its ``QuestionID``. This is what
:func:`kli.survey_tree.verify_roundtrip` checks, and it is the structural fix
for the silent answer-misassignment hazard described in
:mod:`kli.survey_tree`. Do not remove it from the skeleton.

**Chunking.** A full survey can run to 60+ questions, which together with a
full persona will exceed a local model's usable context and degrade its
answers well before it overflows. :func:`chunk_refs` splits the question set so
a study can be run in several smaller calls per participant. Chunk boundaries
never split a question.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any, Iterable, Sequence

from kli.survey_tree import QuestionRef

__all__ = [
    "clean_text",
    "describe_question",
    "answer_skeleton",
    "build_question_prompt",
    "chunk_refs",
    "UnsupportedQuestionType",
]


class UnsupportedQuestionType(ValueError):
    """Raised for a question type this renderer cannot represent.

    Add support in three places (see README): a branch in
    :func:`describe_question`, a branch in :func:`answer_skeleton`, and a
    validator in :mod:`kli.parse_responses`.
    """


_TAG_RE = re.compile(r"<[^>]*>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")


def clean_text(value: Any) -> str:
    """Strip HTML tags and unescape entities, preserving paragraph breaks.

    The vendored ``strip_html`` in ``text_simulation/convert_persona_to_text.py``
    collapses all whitespace (including newlines) and does not unescape HTML
    entities beyond ``&nbsp;``, so ``&rsquo;`` and ``&mdash;`` leak into prompts
    verbatim. It also joins adjacent inline tags without a separator, turning
    ``<b>BBC</b><i>News</i>`` into ``BBCNews``. This fixes all three.

    Args:
        value: Raw question text, possibly containing HTML, or any scalar.

    Returns:
        Cleaned plain text.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)

    # Block-level tags become newlines; inline tags become a space, so adjacent
    # inline elements do not run their text together.
    text = re.sub(r"(?i)<\s*(br|/p|/div|/tr|/li|/h[1-6])\s*/?\s*>", "\n", value)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = _WS_RE.sub(" ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def _selector(ref: QuestionRef) -> str:
    return str(ref.question.get("Settings", {}).get("Selector", "") or "")


def _is_multi_select(ref: QuestionRef) -> bool:
    return _selector(ref).upper() in {"MAVR", "MAHR", "MACOL"}


def _options(ref: QuestionRef) -> list[str]:
    return [clean_text(o) for o in ref.question.get("Options", []) or []]


def _rows(ref: QuestionRef) -> list[str]:
    return [clean_text(r) for r in ref.question.get("Rows", []) or []]


def _columns(ref: QuestionRef) -> list[str]:
    return [clean_text(c) for c in ref.question.get("Columns", []) or []]


def _statements(ref: QuestionRef) -> list[str]:
    return [clean_text(s) for s in ref.question.get("Statements", []) or []]


def _numbered(items: Sequence[str], *, bullet: str = "  ") -> str:
    return "\n".join(
        f"{bullet}{i} = {item or '(no label)'}" for i, item in enumerate(items, 1)
    )


def describe_question(ref: QuestionRef, *, include_reasoning: bool = False) -> str:
    """Render one question as prompt text.

    Args:
        ref: The question to render.
        include_reasoning: Unused here; reasoning is requested via the response
            skeleton, not the question text. Accepted for symmetry.

    Returns:
        The prompt block for this question, without a trailing newline.

    Raises:
        UnsupportedQuestionType: For a type with no renderer.
    """
    text = clean_text(ref.question.get("QuestionText", ""))

    if ref.is_descriptive:
        return f"[Information shown to the respondent]\n{text}" if text else ""

    header = f"{ref.label}. [QuestionID: {ref.question_id}]"
    parts = [header]
    if text:
        parts.append(text)

    qtype = ref.question_type

    if qtype == "MC":
        options = _options(ref)
        if _is_multi_select(ref):
            parts.append("Question type: Multiple choice (select ALL that apply)")
        else:
            parts.append("Question type: Single choice (select EXACTLY ONE)")
        if options:
            parts.append("Options:\n" + _numbered(options))

    elif qtype == "Matrix":
        rows, columns = _rows(ref), _columns(ref)
        parts.append(
            "Question type: Matrix — rate EVERY item below on the same scale"
        )
        if columns:
            parts.append("Scale:\n" + _numbered(columns))
        if rows:
            parts.append("Items to rate (answer all, in order):\n" + _numbered(rows))

    elif qtype == "Slider":
        statements = _statements(ref)
        rng = ref.question.get("Range", {}) or {}
        lo, hi = rng.get("Min"), rng.get("Max")
        bounds = f" between {lo} and {hi}" if lo is not None and hi is not None else ""
        parts.append(f"Question type: Slider — give a number{bounds} for each item")
        if statements:
            labelled = [s or "(overall)" for s in statements]
            parts.append("Items (answer all, in order):\n" + _numbered(labelled))

    elif qtype == "CS":
        rows = _rows(ref)
        settings = ref.question.get("Settings", {}) or {}
        total = settings.get("ChoiceTotal")
        total_note = f" The values MUST sum to exactly {total}." if total else ""
        parts.append(
            "Question type: Constant sum — distribute points across the items "
            f"below.{total_note}"
        )
        if rows:
            parts.append("Items (answer all, in order):\n" + _numbered(rows))

    elif qtype == "TE":
        selector = _selector(ref).upper()
        if selector == "FORM":
            rows = _rows(ref)
            parts.append("Question type: Text entry form — fill in every field")
            if rows:
                parts.append("Fields (answer all, in order):\n" + _numbered(rows))
            defaults = ref.question.get("DefaultChoices") or []
            given = [
                f"  {i} = {clean_text(d)}"
                for i, d in enumerate(defaults, 1)
                if d is not None
            ]
            if given:
                parts.append("Fields already filled in (keep these values):\n" + "\n".join(given))
        elif selector == "ESTB":
            parts.append("Question type: Numeric entry — answer with a number only")
        else:
            parts.append("Question type: Text entry — answer in your own words")

    else:
        raise UnsupportedQuestionType(
            f"No renderer for QuestionType {qtype!r} "
            f"({ref.question_id}, selector {_selector(ref)!r}). "
            "Add a branch here, a branch in answer_skeleton(), and a validator "
            "in kli/parse_responses.py. Alternatively list this QuestionID under "
            "'exclude_questions' in the study config."
        )

    return "\n".join(p for p in parts if p)


def answer_skeleton(ref: QuestionRef, *, include_reasoning: bool = False) -> dict[str, Any]:
    """Build the expected ``Answers`` skeleton for one question, with nulls.

    The shapes mirror what ``processing_qualtrics_csv/`` writes for human
    respondents, so twin and human records are directly comparable.

    Args:
        ref: The question.
        include_reasoning: If True, add a ``"Reasoning"`` field for the model to
            fill before answering.

    Returns:
        A JSON-serializable dict with ``QuestionID``, optional ``Reasoning``,
        and ``Answers``.

    Raises:
        UnsupportedQuestionType: For a type with no skeleton.
    """
    qtype = ref.question_type
    answers: dict[str, Any]

    if qtype == "MC":
        if _is_multi_select(ref):
            answers = {"SelectedByPosition": [], "SelectedText": []}
        else:
            answers = {"SelectedByPosition": None, "SelectedText": None}

    elif qtype == "Matrix":
        n = len(_rows(ref)) or 1
        answers = {
            "SelectedByPosition": [None] * n,
            "SelectedText": [None] * n,
        }

    elif qtype == "Slider":
        n = len(_statements(ref)) or 1
        answers = {"Values": [None] * n}

    elif qtype == "CS":
        n = len(_rows(ref)) or 1
        answers = {"Values": [None] * n}

    elif qtype == "TE":
        if _selector(ref).upper() == "FORM":
            n = len(_rows(ref)) or 1
            answers = {"Text": [None] * n}
        else:
            answers = {"Text": None}

    else:
        raise UnsupportedQuestionType(
            f"No answer skeleton for QuestionType {qtype!r} ({ref.question_id})."
        )

    entry: dict[str, Any] = {"QuestionID": ref.question_id}
    if include_reasoning:
        entry["Reasoning"] = None
    entry["Answers"] = answers
    return entry


_OPENING_WITH_PERSONA = (
    "Answer every question below as the person described above would answer it."
)

#: Control-arm opening. The persona-arm wording ("as the person described
#: above") is incoherent when no persona is supplied, and a model told to
#: imitate someone it was never shown is a confound rather than a clean
#: baseline. The control therefore asks for a plain answer. This is the ONLY
#: wording difference between the arms, and
#: :func:`kli.assemble_prompts.check_arm_consistency` plus the run manifest
#: both record it so it can be stated in a methods section.
_OPENING_NO_PERSONA = "Answer every question below."

_INSTRUCTIONS = """\
{opening}

Return ONE JSON object and nothing else. Use the template at the end of this
message exactly as given: keep every key, keep the same order, and replace each
null with your answer. Do not add keys. Do not remove keys. Do not wrap the
object in any other structure.

Rules for answer values:
- "SelectedByPosition" is the NUMBER of the option you chose, from the numbered
  list shown for that question. "SelectedText" is that option's exact text.
- Where a value is a list, return a list of the SAME LENGTH as the template,
  with one entry per item, in the order the items were listed.
- For single choice, "SelectedByPosition" is one number, not a list.
- For multiple choice, return every option you would select.
- For constant sum questions, the numbers must add up to the stated total.
- Answer with numbers where the template shows a number, and text where it
  shows text. Never leave a null in your reply.
- Keep each "QuestionID" exactly as given. It is how your answer is matched to
  the question.
"""

_REASONING_NOTE = """\
- Fill "Reasoning" with one short sentence explaining the answer, before you
  give the answer itself.
"""


def build_question_prompt(
    refs: Iterable[QuestionRef],
    *,
    include_reasoning: bool = False,
    persona_present: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Render a set of questions into a prompt plus the expected skeleton.

    Args:
        refs: Questions in canonical order, from
            :func:`kli.survey_tree.walk_questions`. Descriptive elements are
            rendered as context and given no answer slot.
        include_reasoning: Request a one-sentence rationale per question.
        persona_present: False in the no-persona control arm. Only the opening
            sentence changes; every question, option list and skeleton entry is
            byte-identical across arms, which is what makes the two runs
            comparable. See :data:`_OPENING_NO_PERSONA`.

    Returns:
        ``(prompt_text, skeleton)`` where ``skeleton`` maps each ``Qn`` label to
        its expected entry. Keep the skeleton: :mod:`kli.parse_responses` uses
        it to validate shape, and it is what makes the round-trip check possible.
    """
    refs = list(refs)
    blocks: list[str] = []
    skeleton: dict[str, Any] = {}

    for ref in refs:
        rendered = describe_question(ref, include_reasoning=include_reasoning)
        if rendered:
            blocks.append(rendered)
        if not ref.is_descriptive:
            skeleton[ref.label] = answer_skeleton(
                ref, include_reasoning=include_reasoning
            )

    instructions = _INSTRUCTIONS.format(
        opening=_OPENING_WITH_PERSONA if persona_present else _OPENING_NO_PERSONA
    )
    if include_reasoning:
        instructions += _REASONING_NOTE

    prompt = (
        instructions
        + "\n"
        + "\n\n".join(blocks)
        + "\n\n--- Reply with this JSON, nulls replaced: ---\n"
        + json.dumps(skeleton, indent=2, ensure_ascii=False)
    )
    return prompt, skeleton


def chunk_refs(
    refs: Sequence[QuestionRef],
    questions_per_call: int | None,
) -> list[list[QuestionRef]]:
    """Split questions into chunks for separate model calls.

    A local 7B-class model given a full persona plus 60 questions will usually
    degrade -- dropping items, repeating answers -- well before it exceeds its
    context window. Chunking trades more calls for better answers.

    Descriptive elements are attached to the chunk containing the next
    answerable question, so context is never separated from the question it
    sets up. Chunk boundaries never split a question.

    Args:
        refs: All questions in canonical order.
        questions_per_call: Answerable questions per chunk. ``None`` or a value
            below 1 means one chunk containing everything (upstream behavior).

    Returns:
        List of chunks, each a list of :class:`QuestionRef`.
    """
    refs = list(refs)
    if not questions_per_call or questions_per_call < 1:
        return [refs] if refs else []

    chunks: list[list[QuestionRef]] = []
    current: list[QuestionRef] = []
    answerable_in_current = 0

    for ref in refs:
        current.append(ref)
        if not ref.is_descriptive:
            answerable_in_current += 1
            if answerable_in_current >= questions_per_call:
                chunks.append(current)
                current = []
                answerable_in_current = 0

    if any(not r.is_descriptive for r in current):
        chunks.append(current)
    elif current and chunks:
        # Trailing descriptive-only text: keep it with the previous chunk rather
        # than emitting a chunk with no answer slots.
        chunks[-1].extend(current)

    return chunks
