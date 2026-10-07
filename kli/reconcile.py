"""Validate Qualtrics ingestion: report what was dropped, and why.

ORIGINAL WORK (Kravis Leadership Institute). See ATTRIBUTION.md.

-------------------------------------------------------------------------------
WHY THIS IS THE FIRST THING TO RUN
-------------------------------------------------------------------------------

The vendored Qualtrics subsystems have no test coverage: no ``tests/``, no
schema validation of outputs, no reconciliation that every CSV column was
consumed, and no report of dropped questions. ``save_output`` is a bare
``json.dump``. The only test in either upstream repo for this layer is a
six-case self-check inside ``logic_eval.py``, and all six exercise
``EmbeddedField`` -- which is precisely why the display-logic defect below went
unnoticed.

The defects that matter most are silent. In particular,
``processing_qualtrics_csv/get_responses.py:139`` calls ``eval_display_logic``
without the ``response_data`` argument, so **every question whose display logic
depends on a previous answer evaluates to False and is dropped from every
participant's file** -- no error, no warning. Two upstream study configs carry
hand-written workarounds for it.

This module makes that visible. Run it after stage 2 and before building any
prompts:

    python -m kli.reconcile --config configs/my_study.yaml

It compares the parsed survey template against the per-participant response
JSONs and reports, per question, how many participants retained it. A question
present in the template but absent for everyone is the display-logic signature.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kli.survey_tree import SKIPPED_TYPES, walk_questions

__all__ = ["Reconciliation", "reconcile", "main"]


@dataclass
class Reconciliation:
    """Ingestion reconciliation results.

    Attributes:
        n_participants: Response files examined.
        template_questions: ``{question_id: question_type}`` from the template.
        presence: ``{question_id: count}`` of participants retaining it.
        answered: ``{question_id: count}`` of participants with an answer.
        display_logic_questions: Template question IDs carrying DisplayLogic.
        type_counts: Question types seen in the template.
    """

    n_participants: int = 0
    template_questions: dict[str, str] = field(default_factory=dict)
    presence: Counter = field(default_factory=Counter)
    answered: Counter = field(default_factory=Counter)
    display_logic_questions: set[str] = field(default_factory=set)
    type_counts: Counter = field(default_factory=Counter)

    @property
    def always_missing(self) -> list[str]:
        """Template questions retained for no participant."""
        return sorted(
            qid for qid in self.template_questions if self.presence[qid] == 0
        )

    @property
    def sometimes_missing(self) -> list[str]:
        """Questions retained for some but not all participants."""
        return sorted(
            qid
            for qid in self.template_questions
            if 0 < self.presence[qid] < self.n_participants
        )

    @property
    def present_but_unanswered(self) -> list[str]:
        """Questions kept in the file but carrying no answer for anyone."""
        return sorted(
            qid
            for qid in self.template_questions
            if self.presence[qid] > 0 and self.answered[qid] == 0
        )

    @property
    def suspected_display_logic_loss(self) -> list[str]:
        """Always-missing questions that carry display logic.

        These are the signature of the upstream display-logic defect: the
        template has them, they are conditional, and no participant retained
        them.
        """
        return sorted(set(self.always_missing) & self.display_logic_questions)

    def to_dict(self) -> dict[str, Any]:
        """Serializable form, for writing alongside the study outputs."""
        return {
            "participants": self.n_participants,
            "template_questions": len(self.template_questions),
            "template_question_types": dict(self.type_counts.most_common()),
            "questions_with_display_logic": len(self.display_logic_questions),
            "always_missing": self.always_missing,
            "suspected_display_logic_loss": self.suspected_display_logic_loss,
            "sometimes_missing": {
                qid: self.presence[qid] for qid in self.sometimes_missing
            },
            "present_but_unanswered": self.present_but_unanswered,
        }

    def summary(self) -> str:
        """Human-readable report."""
        lines = [
            f"participants examined        : {self.n_participants}",
            f"questions in survey template : {len(self.template_questions)}",
            f"  with display logic         : {len(self.display_logic_questions)}",
            "",
            f"never present in any file    : {len(self.always_missing)}",
            f"present for some, not all    : {len(self.sometimes_missing)}",
            f"present but never answered   : {len(self.present_but_unanswered)}",
        ]

        suspected = self.suspected_display_logic_loss
        if suspected:
            lines += [
                "",
                "!! LIKELY SILENT DATA LOSS !!",
                f"{len(suspected)} question(s) carry display logic and were "
                "dropped for EVERY participant:",
                "  " + ", ".join(suspected[:20])
                + (" ..." if len(suspected) > 20 else ""),
                "",
                "This is the known upstream defect: get_responses.py:139 calls",
                "eval_display_logic() without response_data, so any condition of",
                "type 'Question' evaluates False and the question is removed.",
                "",
                "If these questions carry variables you need, work around it by",
                "nulling their DisplayLogic and keeping them explicitly:",
                "",
                "  qsf_to_json:",
                "    revise_questions:",
                f"      {suspected[0]}: {{DisplayLogic: null}}",
                "  csv_to_fill_answers:",
                "    keep_questions_when_answer_missing:",
                f"      - {suspected[0]}",
                "",
                "then re-run stages 1 and 2 and re-run this check.",
            ]
        elif self.always_missing:
            lines += [
                "",
                f"{len(self.always_missing)} template question(s) never appear, "
                "none of them conditional.",
                "Usually excluded blocks or unsupported types -- worth a look:",
                "  " + ", ".join(self.always_missing[:20]),
            ]
        else:
            lines += ["", "No always-missing questions. Ingestion looks clean."]

        return "\n".join(lines)


def _template_questions(template: Any) -> tuple[dict[str, str], set[str], Counter]:
    """Collect question IDs, types, and which carry display logic."""
    questions: dict[str, str] = {}
    with_logic: set[str] = set()
    types: Counter = Counter()

    def walk(elements: list[dict[str, Any]]) -> None:
        for element in elements:
            if not isinstance(element, dict):
                continue
            if element.get("ElementType") == "Block":
                for question in element.get("Questions", []) or []:
                    if not isinstance(question, dict):
                        continue
                    qid = question.get("QuestionID")
                    qtype = question.get("QuestionType", "")
                    if not qid or qtype in SKIPPED_TYPES:
                        continue
                    questions[qid] = qtype
                    types[qtype] += 1
                    if question.get("DisplayLogic"):
                        with_logic.add(qid)
            if "Elements" in element:
                walk(element["Elements"])

    root = template.get("Elements", []) if isinstance(template, dict) else template
    walk(root)
    return questions, with_logic, types


def reconcile(
    *, survey_template_path: str | Path, response_json_dir: str | Path
) -> Reconciliation:
    """Compare the survey template against the per-participant response files.

    Args:
        survey_template_path: Output of ``parse_qsf.py``.
        response_json_dir: Output directory of ``get_responses.py``.

    Returns:
        A :class:`Reconciliation`.

    Raises:
        FileNotFoundError: If the template or any response files are missing.
    """
    survey_template_path = Path(survey_template_path)
    response_json_dir = Path(response_json_dir)

    if not survey_template_path.is_file():
        raise FileNotFoundError(
            f"Survey template not found: {survey_template_path}. "
            "Run processing_qualtrics_qsf/parse_qsf.py first."
        )
    paths = sorted(response_json_dir.glob("pid_*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No pid_*.json files in {response_json_dir}. "
            "Run processing_qualtrics_csv/get_responses.py first."
        )

    template = json.loads(survey_template_path.read_text(encoding="utf-8"))
    questions, with_logic, types = _template_questions(template)

    result = Reconciliation(
        template_questions=questions,
        display_logic_questions=with_logic,
        type_counts=types,
    )

    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        result.n_participants += 1
        for ref in walk_questions(document):
            result.presence[ref.question_id] += 1
            answers = ref.question.get("Answers")
            if isinstance(answers, dict) and answers:
                result.answered[ref.question_id] += 1

    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        0 if ingestion looks clean, 1 if suspected silent data loss was found --
        so this can gate a pipeline step in CI or a shell script.
    """
    from kli.config import get, load_config

    parser = argparse.ArgumentParser(
        description="Check Qualtrics ingestion for silently dropped questions."
    )
    parser.add_argument("--config", required=True, help="Study YAML config.")
    parser.add_argument(
        "--json-out", default=None, help="Optional path to write the full report."
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    result = reconcile(
        survey_template_path=get(config, "qsf_to_json.output_file"),
        response_json_dir=get(config, "csv_to_fill_answers.output_dir"),
    )

    print(result.summary())

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"\nfull report written to {out}")

    return 1 if result.suspected_display_logic_loss else 0


if __name__ == "__main__":
    raise SystemExit(main())
