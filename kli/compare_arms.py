"""Compare twin arms against each other: the tables behind H1-H3.

ORIGINAL WORK (Kravis Leadership Institute). See ATTRIBUTION.md.

:mod:`kli.pilot_metrics` scores one arm at a time. The study's hypotheses are
contrasts *between* arms on the same participants and the same holdout items,
for example "the meso twin agrees with its human more often than the same
participant's no-persona twin". This module runs the pilot metrics on every arm
named in the config and lays the arms side by side.

Everything here is descriptive. With three participants the contrasts are
reported, not tested: the output gives each participant's difference and how
many participants moved in the predicted direction, and nothing more.

-------------------------------------------------------------------------------
WHAT IS COMPUTED
-------------------------------------------------------------------------------

Per arm (``arm_summary.csv``):
    exact-match rate and mean absolute deviation (MAD) over the holdout items,
    on each twin's first replicate, as :mod:`kli.pilot_metrics` does; and the
    run-to-run modal-answer rate when the arm has two or more replicates.
    Stability is reported beside agreement because it bounds it.

Per contrast and participant (``contrasts.csv``):
    focal minus comparator, for both exact-match rate and MAD. A hypothesis of
    the form "the focal twin does better" predicts a positive exact-match
    difference and a negative MAD difference; ``supports_agreement`` and
    ``supports_mad`` record whether each participant went that way.

Every arm must cover the same participants and the same items, or the
comparison is refused: a contrast over different item sets is not a contrast.

Usage::

    python -m kli.compare_arms --config configs/study/3_compare_arms.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from kli.config import get, load_config
from kli.pilot_metrics import (
    compute_item_agreement,
    compute_stability,
    extract_item_values,
    load_humans,
    load_replicates,
)

__all__ = ["compare_arms", "main"]


def _score_arm(name: str, directory: Path, humans: dict[str, Any]) -> dict[str, Any]:
    replicates = load_replicates(directory)
    if not replicates:
        raise FileNotFoundError(f"arm {name!r}: no pid_*.json files in {directory}")
    first_run = {pid: docs[0] for pid, docs in replicates.items()}
    agreement = compute_item_agreement(humans, first_run)
    n_reps = min(len(docs) for docs in replicates.values())
    stability = compute_stability(replicates) if n_reps >= 2 else None
    return {
        "participants": sorted(replicates),
        "items": {
            pid: sorted(extract_item_values(doc)) for pid, doc in first_run.items()
        },
        "agreement": agreement,
        "summary": {
            "arm": name,
            "participants": len(replicates),
            "replicates": n_reps,
            "exact_match_rate": agreement.summary["exact_match_rate"],
            "mean_abs_deviation": agreement.summary["mean_abs_deviation"],
            "modal_rate": stability.summary["modal_rate"] if stability else float("nan"),
        },
    }


def compare_arms(
    *,
    human_dir: str | Path,
    arms: dict[str, str | Path],
    contrasts: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every arm and every named contrast.

    Args:
        human_dir: The participants' own answers to the holdout items.
        arms: Arm name to that arm's imputed response directory.
        contrasts: Contrast name (e.g. ``"H2"``) to
            ``{"focal": <arm>, "comparator": <arm>}``.

    Returns:
        ``(arm_summary, contrast_table)``.

    Raises:
        ValueError: If a contrast names an unknown arm, or arms differ in
            participants or items.
    """
    for label, pair in contrasts.items():
        for role in ("focal", "comparator"):
            if pair.get(role) not in arms:
                raise ValueError(
                    f"contrast {label}: {role} {pair.get(role)!r} is not one of the "
                    f"arms {sorted(arms)}"
                )

    humans = load_humans(human_dir)
    if not humans:
        raise FileNotFoundError(f"no pid_*.json files in {human_dir}")
    scored = {name: _score_arm(name, Path(d), humans) for name, d in arms.items()}

    reference_name = next(iter(scored))
    reference = scored[reference_name]
    for name, arm in scored.items():
        if arm["participants"] != reference["participants"]:
            raise ValueError(
                f"arm {name!r} covers participants {arm['participants']}, but "
                f"{reference_name!r} covers {reference['participants']}. Every "
                "arm must be run on the same participants."
            )
        if arm["items"] != reference["items"]:
            raise ValueError(
                f"arm {name!r} answered different items from {reference_name!r}. "
                "Every arm must answer the same holdout items."
            )

    summary = pd.DataFrame([arm["summary"] for arm in scored.values()])

    rows: list[dict[str, Any]] = []
    for label, pair in contrasts.items():
        focal = scored[pair["focal"]]["agreement"].per_participant.set_index("participant")
        other = scored[pair["comparator"]]["agreement"].per_participant.set_index("participant")
        for pid in focal.index:
            d_match = focal.at[pid, "exact_match_rate"] - other.at[pid, "exact_match_rate"]
            d_mad = focal.at[pid, "mean_abs_deviation"] - other.at[pid, "mean_abs_deviation"]
            rows.append(
                {
                    "contrast": label,
                    "focal": pair["focal"],
                    "comparator": pair["comparator"],
                    "participant": pid,
                    "exact_match_diff": d_match,
                    "mad_diff": d_mad,
                    "supports_agreement": bool(d_match > 0),
                    "supports_mad": bool(d_mad < 0),
                }
            )
    return summary, pd.DataFrame(rows)


def _report(summary: pd.DataFrame, table: pd.DataFrame) -> str:
    lines = ["Arms (first replicate; stability across all replicates)", "=" * 60]
    for _, r in summary.iterrows():
        modal = (
            f"{r['modal_rate']:6.1%}"
            if pd.notna(r["modal_rate"])
            else "   n/a (1 replicate)"
        )
        lines.append(
            f"{r['arm']:<16} match {r['exact_match_rate']:6.1%}   "
            f"MAD {r['mean_abs_deviation']:6.3f}   modal {modal}"
        )
    lines += ["", "Contrasts (focal minus comparator; descriptive only)", "=" * 60]
    for label, group in table.groupby("contrast", sort=False):
        n = len(group)
        lines.append(
            f"{label}: {group['focal'].iat[0]} vs {group['comparator'].iat[0]}  "
            f"match diff {group['exact_match_diff'].mean():+.3f} "
            f"({int(group['supports_agreement'].sum())}/{n} in predicted direction), "
            f"MAD diff {group['mad_diff'].mean():+.3f} "
            f"({int(group['supports_mad'].sum())}/{n})"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="Comparison YAML config.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    summary, table = compare_arms(
        human_dir=get(config, "compare_arms.human_dir"),
        arms=dict(get(config, "compare_arms.arms")),
        contrasts=dict(get(config, "compare_arms.contrasts")),
    )
    print(_report(summary, table))

    out = Path(get(config, "compare_arms.output_dir"))
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "arm_summary.csv", index=False)
    table.to_csv(out / "contrasts.csv", index=False)
    (out / "comparison_config.json").write_text(
        json.dumps(get(config, "compare_arms"), indent=2), encoding="utf-8"
    )
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
