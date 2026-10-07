"""Canonical traversal of a Qualtrics-derived response document.

ORIGINAL WORK (Kravis Leadership Institute). Not derived from either upstream
repository. See ATTRIBUTION.md.

-------------------------------------------------------------------------------
WHY THIS MODULE EXISTS
-------------------------------------------------------------------------------

Two separate problems are solved here.

**1. The schema adapter.**

`processing_qualtrics_csv/get_responses.py` emits, per participant, a document
shaped like::

    {"Metadata": {...}, "Elements": [...], "EmbeddedData": {...}}

whose ``Elements`` may nest ``Block``, ``Branch``, ``Randomizer`` and ``Group``
containers to arbitrary depth. (Measured on real data: ``story_beliefs`` is flat
at depth 0; ``accuracy_nudges`` and ``junk_fees`` reach depth 3 with Branch,
Randomizer and Group wrappers.)

The vendored twin-construction code from the `digital-twin-simulation` repo
assumes a flat, top-level **list** of Blocks -- see
``text_simulation/convert_question_json_to_text.py:281``
(``for element in data: ## data is a list of blocks``) and
``text_simulation/postprocess_responses.py:235``. Handed the dict above it
iterates dict *keys* and raises ``AttributeError``. This module normalizes both
shapes.

**2. The positional-binding hazard.**

This is the more serious problem, and the reason this module is the single
entry point for *both* prompt construction and response parsing.

Upstream, the prompt builder numbers questions ``Q1..Qn`` while walking the
document, and the postprocessor independently re-walks the same document with
its own counter and looks up ``response["Q{count}"]``. There is no question-ID
round trip. If the two traversals ever disagree -- a Randomizer handled
differently, a descriptive block counted on one side but not the other -- then
every answer after the point of divergence is silently attached to the wrong
question. Nothing raises. The output looks entirely normal.

The fix is structural, not defensive: there is exactly one traversal
(:func:`walk_questions`), both sides call it, and the prompt carries each
question's ``QuestionID`` so that :func:`verify_roundtrip` can *prove* the
model's answer was matched to the question it was asked about. A mismatch is an
error, not a silent reassignment.

Do not write a second traversal. If you need a different ordering or filter,
extend this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

__all__ = [
    "QuestionRef",
    "ANSWERABLE_TYPES",
    "DESCRIPTIVE_TYPES",
    "SKIPPED_TYPES",
    "elements_of",
    "walk_questions",
    "question_label",
    "fingerprint",
    "verify_roundtrip",
    "RoundtripError",
    "count_questions",
]

#: Question types that carry an answer we ask the model to produce.
ANSWERABLE_TYPES = frozenset({"MC", "Matrix", "Slider", "TE", "CS"})

#: Types that are shown to the respondent but carry no answer. Rendered into the
#: prompt as context, never given an answer slot.
DESCRIPTIVE_TYPES = frozenset({"DB"})

#: Types present in Qualtrics exports that carry no respondent-meaningful answer
#: and are excluded entirely. ``get_responses.py`` already drops most of these,
#: but they are named here so the decision is explicit rather than incidental.
SKIPPED_TYPES = frozenset({"Timing", "Meta", "Captcha"})

#: Container element types whose children are traversed.
_CONTAINER_TYPES = frozenset({"Branch", "Randomizer", "Group"})


class RoundtripError(RuntimeError):
    """Raised when a model response cannot be matched back to its question.

    This indicates that prompt construction and response parsing disagreed about
    the structure of the survey. Treat it as data corruption, not a transient
    fault: do not retry, and do not fall back to positional matching.
    """


@dataclass(frozen=True)
class QuestionRef:
    """One question located in a response document.

    Attributes:
        ordinal: 1-based position in the canonical walk over *answerable*
            questions. This is the ``n`` in the ``Qn`` label the model sees.
            Descriptive (``DB``) elements do not consume an ordinal.
        question_id: The Qualtrics ``QuestionID`` (e.g. ``"QID1218539715"``).
            Loop-and-merge questions carry a prefixed form (``"1_QID34"``).
        question: The live question dict. Mutating it mutates the document,
            which is how imputed answers are written back.
        path: Structural breadcrumb to this question, for error messages.
            e.g. ``("Randomizer:FL_90", "Block:ABeliefs1", "QID1218539715")``.
        question_type: The Qualtrics ``QuestionType``.
        is_descriptive: True for ``DB`` blocks of text with no answer slot.
    """

    ordinal: int
    question_id: str
    question: dict[str, Any]
    path: tuple[str, ...]
    question_type: str
    is_descriptive: bool

    @property
    def label(self) -> str:
        """The ``Qn`` key used in the prompt and the model's JSON response."""
        return f"Q{self.ordinal}"


def elements_of(document: Any) -> list[dict[str, Any]]:
    """Normalize a response document to its top-level element list.

    Accepts every shape observed across the two upstream repositories:

    * ``{"Metadata": ..., "Elements": [...]}`` -- what
      ``processing_qualtrics_csv/get_responses.py`` writes.
    * ``[...]`` -- a bare list of Blocks, which is what the HuggingFace
      ``answer_blocks`` files contain and what the vendored demo code expects.
    * A JSON string wrapping either of the above. The HuggingFace persona
      columns are sometimes stored double-encoded.

    Args:
        document: The parsed (or string) response document.

    Returns:
        The list of top-level elements.

    Raises:
        TypeError: If the document is neither a list nor a dict with
            ``Elements``.
    """
    if isinstance(document, str):
        import json

        text = document.strip()
        # Tolerate the double-encoded form seen in some HF persona columns.
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].replace('\\"', '"')
        document = json.loads(text)

    if isinstance(document, list):
        return document
    if isinstance(document, dict):
        if "Elements" in document:
            elements = document["Elements"]
            if not isinstance(elements, list):
                raise TypeError(
                    f"'Elements' must be a list, got {type(elements).__name__}"
                )
            return elements
        raise TypeError(
            "Response document is a dict without an 'Elements' key. "
            f"Keys present: {sorted(document)}. This does not look like output "
            "from processing_qualtrics_csv/get_responses.py."
        )
    raise TypeError(
        f"Cannot read a response document of type {type(document).__name__}; "
        "expected a list of Blocks or a dict with an 'Elements' key."
    )


def walk_questions(
    document: Any,
    *,
    include_descriptive: bool = True,
) -> Iterator[QuestionRef]:
    """Walk a response document in canonical order, yielding every question.

    This is **the** traversal. Prompt construction and response parsing must
    both use it, so that ``Qn`` labels mean the same thing on both sides. See
    the module docstring for why.

    Order is document order, depth-first through ``Branch``, ``Randomizer`` and
    ``Group`` containers. It is fully deterministic for a given document: no
    set iteration, no dict ordering dependence, no randomness.

    Ordinals are assigned only to answerable questions, so adding or removing a
    descriptive block does not renumber the answer slots.

    Args:
        document: A response document in any shape :func:`elements_of` accepts.
        include_descriptive: If True (default), ``DB`` elements are yielded with
            ``is_descriptive=True`` and ``ordinal=0`` so they can be rendered
            into the prompt as context. If False, they are omitted entirely.

    Yields:
        :class:`QuestionRef` for each question, in canonical order.
    """
    counter = 0

    def _walk(
        elements: list[dict[str, Any]], path: tuple[str, ...]
    ) -> Iterator[QuestionRef]:
        nonlocal counter
        for element in elements:
            if not isinstance(element, dict):
                continue
            element_type = element.get("ElementType")

            if element_type == "Block":
                block_name = element.get("BlockName") or element.get("BlockID") or "?"
                block_path = (*path, f"Block:{block_name}")
                for question in element.get("Questions", []):
                    if not isinstance(question, dict):
                        continue
                    qid = question.get("QuestionID")
                    qtype = question.get("QuestionType", "")
                    if qid is None or qtype in SKIPPED_TYPES:
                        continue

                    if qtype in DESCRIPTIVE_TYPES:
                        if include_descriptive:
                            yield QuestionRef(
                                ordinal=0,
                                question_id=qid,
                                question=question,
                                path=(*block_path, qid),
                                question_type=qtype,
                                is_descriptive=True,
                            )
                        continue

                    counter += 1
                    yield QuestionRef(
                        ordinal=counter,
                        question_id=qid,
                        question=question,
                        path=(*block_path, qid),
                        question_type=qtype,
                        is_descriptive=False,
                    )

            elif element_type in _CONTAINER_TYPES:
                marker = element.get("FlowID") or element.get("Description") or "?"
                yield from _walk(
                    element.get("Elements", []), (*path, f"{element_type}:{marker}")
                )

            elif "Elements" in element:
                # Unknown container type. Traverse rather than drop: dropping it
                # would silently lose every question beneath it, and a changed
                # ordinal count is exactly the failure this module exists to
                # prevent. Upstream drops these.
                marker = element.get("FlowID") or "?"
                yield from _walk(
                    element["Elements"], (*path, f"{element_type or 'Unknown'}:{marker}")
                )

    yield from _walk(elements_of(document), ())


def question_label(ordinal: int) -> str:
    """Return the prompt/response key for a given ordinal."""
    return f"Q{ordinal}"


def count_questions(document: Any) -> int:
    """Count answerable questions in a document."""
    return sum(1 for _ in walk_questions(document, include_descriptive=False))


def fingerprint(document: Any) -> tuple[tuple[int, str], ...]:
    """Return the ordered ``(ordinal, question_id)`` pairs for a document.

    Two documents with the same fingerprint will produce identically-labelled
    prompts. Comparing fingerprints before and after a round trip is the cheap
    way to prove that prompt construction and parsing agree.

    Args:
        document: A response document.

    Returns:
        Tuple of ``(ordinal, question_id)``, in canonical order.
    """
    return tuple(
        (ref.ordinal, ref.question_id)
        for ref in walk_questions(document, include_descriptive=False)
    )


def verify_roundtrip(
    ref: QuestionRef,
    response_entry: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    """Check that a model response entry belongs to the question we asked.

    The prompt emits each question's ``QuestionID`` alongside its ``Qn`` label,
    and asks the model to echo it back. This verifies the echo.

    Upstream does not do this, which is why a traversal mismatch there silently
    misassigns answers instead of failing.

    Args:
        ref: The question the prompt asked about.
        response_entry: The model's entry for ``ref.label``.
        strict: If True, a missing ``QuestionID`` in the response is an error.
            If False, a missing ID is tolerated (the model omitted it) but a
            *wrong* ID is still an error.

    Raises:
        RoundtripError: If the echoed ``QuestionID`` is absent (when strict) or
            does not match.
    """
    echoed = response_entry.get("QuestionID") or response_entry.get("question_id")

    if echoed is None:
        if strict:
            raise RoundtripError(
                f"{ref.label} ({ref.question_id}) : model response omitted "
                "'QuestionID', so the answer cannot be verified as belonging to "
                "this question. Path: " + " > ".join(ref.path)
            )
        return

    if str(echoed).strip() != str(ref.question_id).strip():
        raise RoundtripError(
            f"{ref.label} : asked about {ref.question_id!r} but the model echoed "
            f"{echoed!r}. Prompt construction and parsing disagree about survey "
            "structure, or the model reordered its output. Answers after this "
            "point cannot be trusted. Path: " + " > ".join(ref.path)
        )
