"""Split the SJTs into holdouts and training items, for the SJT-inclusion arm.

ORIGINAL WORK (Kravis Leadership Institute). See ATTRIBUTION.md.

-------------------------------------------------------------------------------
THE DESIGN
-------------------------------------------------------------------------------

Wave 3 contains ten situational judgment tests (``SJT1``-``SJT10``,
``QID1101``-``QID1110``): a scenario followed by six actions, each rated 1-7 for
how likely the respondent is to take it.

* **Two SJTs are holdouts in every arm.** They are drawn at random, once for the
  whole study, from a recorded seed. Every twin in every arm is asked those two
  and nothing else, and is scored on them. Because the test set never changes
  between arms, any difference between arms is a difference in what the twin
  was given, not in what it was asked.

* **The other eight are training SJTs.** The switch
  ``include_training_sjts_in_persona`` decides whether a twin's prompt also
  contains the participant's own answers to them. When it is off they are not
  used at all.

Why one draw for the whole study rather than one per participant. With a
per-participant draw, the scrambled arm breaks: participant A's donor B has
answered A's holdout scenarios in B's training block, so the scrambled twin would
see a rating of the very scenario it is about to be scored on while the matched
twin would not. The two arms would then differ in more than the pairing. A single
draw keeps the training and holdout sets identical for everyone.

Why at random. The SJTs are shown in fixed order, so "the last two" would
confound holdout status with position and fatigue. A seeded draw is defensible
and reproducible: the seed and the result are written to ``sjt_split.json`` and
to every run manifest.

-------------------------------------------------------------------------------
WHAT ``prepare`` WRITES
-------------------------------------------------------------------------------

From the combined three-wave documents (:mod:`kli.combine_waves`)::

    <output_dir>/persona_json/    all waves, ALL TEN SJTs removed -> persona base
    <output_dir>/holdout_json/    the two holdout SJTs only        -> twins answer these
    <output_dir>/training_json/   the eight training SJTs only     -> added when the switch is on
    <output_dir>/sjt_split.json   the draw: seed, holdout and training QIDs

Render ``persona_json`` to text with ``batch_convert_personas.py`` exactly as
before. The training block is added at simulation time by
:func:`render_training_sjts`, so one persona directory serves both settings of
the switch and the persona text is provably identical across them.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from kli.assemble_prompts import extract_pid
from kli.combine_waves import filter_document
from kli.config import get, load_config
from kli.render_questions import clean_text
from kli.survey_tree import SKIPPED_TYPES, walk_questions

__all__ = [
    "KLI_SJT_QUESTION_IDS",
    "DEFAULT_TRAINING_HEADER",
    "SjtSplit",
    "SjtLeakError",
    "draw_split",
    "load_split",
    "split_document",
    "render_training_sjts",
    "check_question_set",
    "prepare",
]

#: SJT1-SJT10 in the KLI Wave 3 survey, in the order shown.
KLI_SJT_QUESTION_IDS = tuple(f"QID{n}" for n in range(1101, 1111))

#: Heading placed above the training SJTs in the prompt.
DEFAULT_TRAINING_HEADER = (
    "## This individual's answers to other workplace scenarios:"
)


class SjtLeakError(ValueError):
    """Raised when a twin would be asked something other than the holdouts.

    Covers both directions: a training SJT among the questions (the twin would
    be scored on an item it may have been shown), and a holdout missing (arms
    would no longer share a test set).
    """


@dataclass(frozen=True)
class SjtSplit:
    """One study-wide assignment of SJTs to holdout and training."""

    sjt_question_ids: tuple[str, ...]
    holdout_question_ids: tuple[str, ...]
    training_question_ids: tuple[str, ...]
    seed: int | None
    method: str

    def manifest(self) -> dict[str, Any]:
        """JSON-ready form, for ``sjt_split.json`` and run manifests."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def draw_split(
    sjt_question_ids: Sequence[str] = KLI_SJT_QUESTION_IDS,
    *,
    n_holdout: int = 2,
    seed: int | None = None,
    holdout_question_ids: Sequence[str] | None = None,
) -> SjtSplit:
    """Choose the holdout SJTs.

    Args:
        sjt_question_ids: Every SJT, in survey order.
        n_holdout: How many to hold out.
        seed: Seed for the random draw. Required unless ``holdout_question_ids``
            is given; an unseeded draw could not be reproduced or reported.
        holdout_question_ids: An explicit choice, overriding the draw.

    Returns:
        The split, with both lists kept in survey order.

    Raises:
        ValueError: On duplicates, an unknown holdout, a missing seed, or a
            holdout count that leaves no training SJTs.
    """
    sjts = tuple(sjt_question_ids)
    if len(set(sjts)) != len(sjts):
        raise ValueError(f"Duplicate SJT question IDs: {sjts}")

    if holdout_question_ids is not None:
        chosen = set(holdout_question_ids)
        unknown = chosen - set(sjts)
        if unknown:
            raise ValueError(f"Holdout IDs not among the SJTs: {sorted(unknown)}")
        method = "explicit"
    else:
        if seed is None:
            raise ValueError(
                "draw_split() needs a seed (holdout_seed in the config) so the "
                "holdout draw can be reproduced and reported."
            )
        if not 0 < n_holdout < len(sjts):
            raise ValueError(
                f"n_holdout must be between 1 and {len(sjts) - 1}, got {n_holdout}"
            )
        chosen = set(random.Random(seed).sample(sjts, n_holdout))
        method = "seeded_random"

    return SjtSplit(
        sjt_question_ids=sjts,
        holdout_question_ids=tuple(q for q in sjts if q in chosen),
        training_question_ids=tuple(q for q in sjts if q not in chosen),
        seed=seed if method == "seeded_random" else None,
        method=method,
    )


def load_split(path: str | Path) -> SjtSplit:
    """Read ``sjt_split.json`` written by :func:`prepare`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SjtSplit(
        sjt_question_ids=tuple(data["sjt_question_ids"]),
        holdout_question_ids=tuple(data["holdout_question_ids"]),
        training_question_ids=tuple(data["training_question_ids"]),
        seed=data.get("seed"),
        method=data.get("method", "unknown"),
    )


def split_document(document: Any, split: SjtSplit) -> dict[str, dict[str, Any]]:
    """Cut one participant's combined document into its three parts.

    Args:
        document: Combined three-wave response document.
        split: The study's split.

    Returns:
        ``{"persona", "holdout", "training"}`` documents. ``persona`` has every
        SJT removed (and timing/meta items, which carry no respondent content).

    Raises:
        SjtLeakError: If any SJT in the split is missing from the document.
            A participant without all ten would silently get a smaller test set
            or a thinner training block.
    """
    present = {ref.question_id for ref in walk_questions(document)}
    missing = [q for q in split.sjt_question_ids if q not in present]
    if missing:
        raise SjtLeakError(f"SJTs missing from this participant's data: {missing}")

    all_sjts = set(split.sjt_question_ids)
    holdout = set(split.holdout_question_ids)
    training = set(split.training_question_ids)
    return {
        "persona": filter_document(
            document,
            lambda q: q.get("QuestionID") not in all_sjts
            and q.get("QuestionType") not in SKIPPED_TYPES,
        ),
        "holdout": filter_document(document, lambda q: q.get("QuestionID") in holdout),
        "training": filter_document(document, lambda q: q.get("QuestionID") in training),
    }


def _render_matrix(question: dict[str, Any]) -> str:
    rows = [clean_text(r) for r in question.get("Rows", []) or []]
    columns = [clean_text(c) for c in question.get("Columns", []) or []]
    selected = (question.get("Answers") or {}).get("SelectedByPosition") or []

    lines = [clean_text(question.get("QuestionText", ""))]
    if columns and any(c != str(i) for i, c in enumerate(columns, 1)):
        lines.append(
            "Scale: " + "; ".join(f"{i} = {c}" for i, c in enumerate(columns, 1))
        )
    elif columns:
        lines.append(f"Scale: 1 to {len(columns)}")
    lines.append("Their ratings:")
    for i, row in enumerate(rows):
        value = selected[i] if i < len(selected) else None
        shown = value if value not in (None, "") else "no answer"
        lines.append(f"  - [{shown}] {row}")
    return "\n".join(lines)


def render_training_sjts(training_document: Any) -> str:
    """Render a participant's training SJTs, with their own ratings, as text.

    Args:
        training_document: The ``training`` part from :func:`split_document`.

    Returns:
        Plain text, one scenario per paragraph. The heading is added by the
        caller so it can be recorded in the manifest.

    Raises:
        ValueError: If an SJT is not a Matrix question. Every KLI SJT is a
            Likert matrix; anything else means the wrong QIDs were configured.
    """
    parts: list[str] = []
    for n, ref in enumerate(walk_questions(training_document, include_descriptive=False), 1):
        if ref.question_type != "Matrix":
            raise ValueError(
                f"{ref.question_id} is {ref.question_type!r}, not a Matrix. "
                "Check sjt_question_ids in the config."
            )
        parts.append(f"Scenario {n}: " + _render_matrix(ref.question))
    return "\n\n".join(parts)


def check_question_set(document: Any, split: SjtSplit) -> None:
    """Refuse to simulate unless the questions are exactly the holdout SJTs.

    Args:
        document: The question document a twin is about to answer.
        split: The study's split.

    Raises:
        SjtLeakError: If any training SJT or non-SJT question is present, or a
            holdout is absent.
    """
    asked = [ref.question_id for ref in walk_questions(document, include_descriptive=False)]
    holdout = set(split.holdout_question_ids)
    leaked = [q for q in asked if q in set(split.training_question_ids)]
    if leaked:
        raise SjtLeakError(
            f"Training SJTs {leaked} are in the question set. The twin would be "
            "scored on items it can be shown. Point question_json_base_dir at "
            "the holdout_json directory written by kli.sjt_split."
        )
    extra = [q for q in asked if q not in holdout]
    missing = [q for q in split.holdout_question_ids if q not in asked]
    if extra or missing:
        raise SjtLeakError(
            f"Question set does not match the SJT holdouts: unexpected {extra}, "
            f"missing {missing}. Every arm must answer exactly the holdouts."
        )


def prepare(
    *,
    combined_json_dir: str | Path,
    output_dir: str | Path,
    split: SjtSplit,
) -> dict[str, Any]:
    """Write persona, holdout and training documents for every participant.

    Args:
        combined_json_dir: Output of :mod:`kli.combine_waves`.
        output_dir: Where the three directories and ``sjt_split.json`` go.
        split: The study's split.

    Returns:
        Summary: participant count and the split.

    Raises:
        FileNotFoundError: If there are no combined documents.
        SjtLeakError: If a participant lacks any SJT.
    """
    output_dir = Path(output_dir)
    paths = [p for p in sorted(Path(combined_json_dir).glob("pid_*.json")) if extract_pid(p)]
    if not paths:
        raise FileNotFoundError(f"No pid_*.json files in {combined_json_dir}")

    for name in ("persona", "holdout", "training"):
        (output_dir / f"{name}_json").mkdir(parents=True, exist_ok=True)

    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        try:
            parts = split_document(document, split)
        except SjtLeakError as exc:
            raise SjtLeakError(f"{path.name}: {exc}") from None
        for name, part in parts.items():
            (output_dir / f"{name}_json" / path.name).write_text(
                json.dumps(part, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    summary = {"participants": len(paths), **split.manifest()}
    (output_dir / "sjt_split.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="Study YAML config.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    block = get(config, "sjt_split")
    split = draw_split(
        block.get("sjt_question_ids") or KLI_SJT_QUESTION_IDS,
        n_holdout=int(block.get("n_holdout", 2)),
        seed=block.get("holdout_seed"),
        holdout_question_ids=block.get("holdout_question_ids"),
    )
    summary = prepare(
        combined_json_dir=get(config, "sjt_split.combined_json_dir"),
        output_dir=get(config, "sjt_split.output_dir"),
        split=split,
    )
    print(
        f"split {summary['participants']} participant(s): holdout "
        f"{summary['holdout_question_ids']} ({summary['method']}, seed "
        f"{summary['seed']}); training {summary['training_question_ids']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
