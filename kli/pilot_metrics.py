"""Item-level agreement and across-run stability, for small-sample pilots.

ORIGINAL WORK (Kravis Leadership Institute). See ATTRIBUTION.md.

-------------------------------------------------------------------------------
WHY A SEPARATE MODULE FROM kli.correspondence
-------------------------------------------------------------------------------

The correspondence metrics in :mod:`kli.correspondence` are computed *across
participants*: a correlation between human and twin scores over the sample, a
standard deviation ratio over the sample, a mean difference over the sample.
With three participants none of those is interpretable. A correlation on n = 3
can be .99 or -.99 by accident, and a standard deviation ratio computed from
three points is almost pure noise. Running them is harmless as a plumbing
check, but reporting them as findings would be misleading.

What *is* measurable with three participants:

**Agreement, item by item.** Did this twin give the same answer as its own
human, on this item? That is a within-pair comparison and does not borrow
strength from the sample. With 3 participants and 40 items you have 120
comparisons, which supports a meaningful agreement rate even though it supports
no correlation.

**Stability across repeated runs.** How much does the same twin, given the same
persona and the same item, vary between runs at a temperature above zero? This
is a property of the model and the prompt, not of the sample, so three
participants estimate it about as well as three hundred. It also bounds
everything else: a twin that disagrees with *itself* on half its answers cannot
agree with a human more often than that, so instability places a ceiling on
achievable agreement.

Both generalize to the full study. Neither becomes redundant once n grows.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from kli.survey_tree import walk_questions

__all__ = [
    "ItemAgreement",
    "StabilityResult",
    "compute_item_agreement",
    "compute_stability",
    "extract_item_values",
    "load_replicates",
    "main",
]

#: Matches the replicate suffix written by kli.run_simulation.
_REP_RE = re.compile(r"_rep(\d+)_response\.json$")


def _scalar(answers: Any, question: dict[str, Any]) -> list[tuple[str, Any]]:
    """Reduce one question's Answers block to ``(sub_item_key, value)`` pairs.

    Multi-part questions (matrix rows, slider statements, constant-sum rows)
    become one entry per part, so that agreement is assessed at the level the
    respondent actually answered rather than on a whole block at once.
    """
    if not isinstance(answers, dict):
        return []

    if "SelectedByPosition" in answers:
        value = answers["SelectedByPosition"]
        if isinstance(value, list):
            return [(str(i + 1), v) for i, v in enumerate(value) if v is not None]
        return [("", value)] if value is not None else []

    if "Values" in answers:
        values = answers["Values"]
        if isinstance(values, list):
            return [(str(i + 1), v) for i, v in enumerate(values) if v is not None]
        return [("", values)] if values is not None else []

    if "Text" in answers:
        text = answers["Text"]
        if isinstance(text, list):
            return [(str(i + 1), v) for i, v in enumerate(text) if v is not None]
        return [("", text)] if text is not None else []

    return []


def extract_item_values(document: Any) -> dict[str, Any]:
    """Flatten a response document to ``{item_key: value}``.

    Item keys are ``QID`` or ``QID_part`` so that a matrix row and its
    neighbours are distinguishable.

    Args:
        document: A response document (human or twin).

    Returns:
        Mapping of item key to answer value.
    """
    values: dict[str, Any] = {}
    for ref in walk_questions(document, include_descriptive=False):
        for part, value in _scalar(ref.question.get("Answers"), ref.question):
            key = f"{ref.question_id}_{part}" if part else ref.question_id
            values[key] = value
    return values


def _numeric(value: Any) -> float:
    """Coerce to float, or NaN."""
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float("nan")
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")


@dataclass
class ItemAgreement:
    """Item-by-item agreement between twins and their own humans.

    Attributes:
        per_item: One row per item: match rate, mean absolute deviation,
            and how many pairs contributed.
        per_participant: One row per participant.
        summary: Scalar summaries.
        n_comparisons: Total human-twin item comparisons made.
    """

    per_item: pd.DataFrame
    per_participant: pd.DataFrame
    summary: dict[str, float]
    n_comparisons: int

    def report(self) -> str:
        """Render a short readable summary."""
        lines = [
            f"Item agreement  ({self.n_comparisons} human-twin item comparisons)",
            "=" * 60,
            f"exact match rate              {self.summary['exact_match_rate']:>8.1%}",
            f"within-one-point rate         {self.summary['within_one_rate']:>8.1%}",
            f"mean absolute deviation       {self.summary['mean_abs_deviation']:>8.3f}",
            f"participants                  {self.summary['n_participants']:>8.0f}",
            f"items                         {self.summary['n_items']:>8.0f}",
        ]
        return "\n".join(lines)


def compute_item_agreement(
    human_docs: dict[str, Any],
    twin_docs: dict[str, Any],
) -> ItemAgreement:
    """Compare each twin to its own human, item by item.

    Args:
        human_docs: ``{participant_id: response document}``.
        twin_docs: Same, for twins. One document per participant; if you have
            replicates, pass one at a time or use :func:`compute_stability`.

    Returns:
        An :class:`ItemAgreement`.

    Raises:
        ValueError: If no participants are shared between the two mappings.
    """
    shared = sorted(set(human_docs) & set(twin_docs))
    if not shared:
        raise ValueError(
            "No participant ids appear in both the human and twin documents."
        )

    rows: list[dict[str, Any]] = []
    for pid in shared:
        human_values = extract_item_values(human_docs[pid])
        twin_values = extract_item_values(twin_docs[pid])
        for key in sorted(set(human_values) & set(twin_values)):
            h_raw, t_raw = human_values[key], twin_values[key]
            h_num, t_num = _numeric(h_raw), _numeric(t_raw)
            numeric = np.isfinite(h_num) and np.isfinite(t_num)
            rows.append(
                {
                    "participant": pid,
                    "item": key,
                    "human": h_raw,
                    "twin": t_raw,
                    "exact_match": (
                        bool(abs(h_num - t_num) < 1e-9)
                        if numeric
                        else str(h_raw).strip().lower() == str(t_raw).strip().lower()
                    ),
                    "abs_deviation": abs(h_num - t_num) if numeric else float("nan"),
                    "numeric": numeric,
                }
            )

    if not rows:
        raise ValueError("No overlapping items between the human and twin documents.")

    frame = pd.DataFrame(rows)

    per_item = (
        frame.groupby("item")
        .agg(
            n_pairs=("exact_match", "size"),
            exact_match_rate=("exact_match", "mean"),
            mean_abs_deviation=("abs_deviation", "mean"),
        )
        .reset_index()
        .sort_values("exact_match_rate")
    )
    per_participant = (
        frame.groupby("participant")
        .agg(
            n_items=("exact_match", "size"),
            exact_match_rate=("exact_match", "mean"),
            mean_abs_deviation=("abs_deviation", "mean"),
        )
        .reset_index()
    )

    numeric_rows = frame[frame["numeric"]]
    summary = {
        "exact_match_rate": float(frame["exact_match"].mean()),
        "within_one_rate": (
            float((numeric_rows["abs_deviation"] <= 1.0).mean())
            if len(numeric_rows)
            else float("nan")
        ),
        "mean_abs_deviation": float(frame["abs_deviation"].mean()),
        "n_participants": float(frame["participant"].nunique()),
        "n_items": float(frame["item"].nunique()),
    }
    return ItemAgreement(
        per_item=per_item,
        per_participant=per_participant,
        summary=summary,
        n_comparisons=len(frame),
    )


@dataclass
class StabilityResult:
    """Variability of a twin's answers across repeated runs.

    Attributes:
        per_item: One row per participant x item: how often the modal answer
            recurred, and the spread across replicates.
        summary: Scalar summaries.
        n_replicates: Replicates per participant.
        warnings: Conditions affecting interpretation.
    """

    per_item: pd.DataFrame
    summary: dict[str, float]
    n_replicates: int
    warnings: list[str] = field(default_factory=list)

    def report(self) -> str:
        """Render a short readable summary."""
        lines = [
            f"Run-to-run stability  ({self.n_replicates} replicates per twin)",
            "=" * 60,
            f"modal-answer rate             {self.summary['modal_rate']:>8.1%}"
            "   (how often the most common answer recurred)",
            f"fully consistent items        {self.summary['fully_consistent_rate']:>8.1%}"
            "   (identical across every run)",
            f"mean SD across runs           {self.summary['mean_sd']:>8.3f}"
            "   (in answer units)",
        ]
        if self.warnings:
            lines.append("")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def compute_stability(replicate_docs: dict[str, list[Any]]) -> StabilityResult:
    """Measure how much each twin's answers move across repeated runs.

    This is the ceiling on achievable agreement: a twin that answers an item
    differently each run cannot match its human more consistently than it
    matches itself.

    Args:
        replicate_docs: ``{participant_id: [doc_rep1, doc_rep2, ...]}``.

    Returns:
        A :class:`StabilityResult`.

    Raises:
        ValueError: If no participant has at least two replicates.
    """
    usable = {pid: docs for pid, docs in replicate_docs.items() if len(docs) >= 2}
    if not usable:
        raise ValueError(
            "Stability needs at least 2 replicates per participant. Set "
            "'replicates' above 1 in the config and re-run."
        )

    rows: list[dict[str, Any]] = []
    for pid, docs in usable.items():
        per_run = [extract_item_values(doc) for doc in docs]
        keys = set(per_run[0])
        for values in per_run[1:]:
            keys &= set(values)
        for key in sorted(keys):
            answers = [values[key] for values in per_run]
            counts: dict[str, int] = defaultdict(int)
            for answer in answers:
                counts[str(answer).strip().lower()] += 1
            modal = max(counts.values())
            numeric = [_numeric(a) for a in answers]
            finite = [v for v in numeric if np.isfinite(v)]
            rows.append(
                {
                    "participant": pid,
                    "item": key,
                    "n_runs": len(answers),
                    "n_distinct": len(counts),
                    "modal_rate": modal / len(answers),
                    "fully_consistent": len(counts) == 1,
                    "sd_across_runs": (
                        float(np.std(finite, ddof=1)) if len(finite) >= 2 else float("nan")
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    n_replicates = int(frame["n_runs"].max()) if len(frame) else 0

    warnings: list[str] = []
    if frame["fully_consistent"].mean() > 0.98:
        warnings.append(
            "Almost every answer is identical across runs. If this run used "
            "temperature 0 that is expected and the stability figure is not "
            "informative; raise the temperature to measure it."
        )
    low = frame[frame["modal_rate"] < 0.6]
    if len(low):
        warnings.append(
            f"{len(low)} of {len(frame)} participant-item pairs had the modal "
            "answer recur in under 60% of runs. Those items are unstable "
            "enough that a single run should not be treated as the twin's "
            "answer."
        )

    summary = {
        "modal_rate": float(frame["modal_rate"].mean()),
        "fully_consistent_rate": float(frame["fully_consistent"].mean()),
        "mean_sd": float(frame["sd_across_runs"].mean()),
        "n_participants": float(frame["participant"].nunique()),
        "n_items": float(frame["item"].nunique()),
    }
    return StabilityResult(
        per_item=frame.sort_values("modal_rate"),
        summary=summary,
        n_replicates=n_replicates,
        warnings=warnings,
    )


def load_replicates(directory: str | Path) -> dict[str, list[Any]]:
    """Load twin documents grouped by participant, across replicates.

    Recognizes both ``pid_<id>_rep<n>_response.json`` and the single-run
    ``pid_<id>_response.json``.

    Args:
        directory: Directory of imputed response JSONs.

    Returns:
        ``{participant_id: [documents in replicate order]}``.
    """
    from kli.assemble_prompts import extract_pid

    directory = Path(directory)
    grouped: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    for path in sorted(directory.glob("pid_*.json")):
        pid = extract_pid(path)
        if pid is None:
            continue
        match = _REP_RE.search(path.name)
        replicate = int(match.group(1)) if match else 1
        try:
            grouped[pid].append((replicate, json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError):
            continue
    return {
        pid: [doc for _, doc in sorted(items, key=lambda pair: pair[0])]
        for pid, items in grouped.items()
    }


def load_humans(directory: str | Path, ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Load human response documents, keyed by participant id.

    Args:
        directory: Directory of per-participant response JSONs.
        ids: Optional subset to load.

    Returns:
        ``{participant_id: document}``.
    """
    from kli.assemble_prompts import extract_pid

    directory = Path(directory)
    wanted = set(ids) if ids is not None else None
    docs: dict[str, Any] = {}
    for path in sorted(directory.glob("pid_*.json")):
        pid = extract_pid(path)
        if pid is None or (wanted is not None and pid not in wanted):
            continue
        try:
            docs[pid] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return docs


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Example::

        python -m kli.pilot_metrics \\
            --human data/pilot/holdout_response_json \\
            --twin  runs/pilot/response_json_llm_imputed \\
            --out   runs/pilot/pilot_metrics
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Item agreement and run-to-run stability for small pilots."
    )
    parser.add_argument("--human", required=True, help="Human response JSON directory.")
    parser.add_argument("--twin", required=True, help="Twin imputed JSON directory.")
    parser.add_argument("--out", default=None, help="Directory for result tables.")
    args = parser.parse_args(argv)

    replicates = load_replicates(args.twin)
    if not replicates:
        parser.error(f"no pid_*.json files found in {args.twin}")
    humans = load_humans(args.human, ids=replicates.keys())

    # Agreement is assessed on the first replicate, so that it describes a
    # single run rather than an average that no run actually produced.
    first_run = {pid: docs[0] for pid, docs in replicates.items()}
    agreement = compute_item_agreement(humans, first_run)
    print(agreement.report())

    stability: StabilityResult | None = None
    if any(len(docs) >= 2 for docs in replicates.values()):
        stability = compute_stability(replicates)
        print()
        print(stability.report())
    else:
        print()
        print(
            "Only one run per twin, so stability was not computed. Set "
            "'replicates' above 1 and raise the temperature above 0 to measure it."
        )

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        agreement.per_item.to_csv(out / "agreement_by_item.csv", index=False)
        agreement.per_participant.to_csv(
            out / "agreement_by_participant.csv", index=False
        )
        if stability is not None:
            stability.per_item.to_csv(out / "stability_by_item.csv", index=False)
        (out / "pilot_summary.json").write_text(
            json.dumps(
                {
                    "agreement": agreement.summary,
                    "n_comparisons": agreement.n_comparisons,
                    "stability": stability.summary if stability else None,
                    "stability_warnings": stability.warnings if stability else [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwritten to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
