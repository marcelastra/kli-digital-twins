"""Human-twin correspondence metrics, following Stojanov (2026).

ORIGINAL WORK (Kravis Leadership Institute). The metric *definitions* are taken
from the published method cited below; the implementation is ours.

Self-contained: depends only on numpy, pandas and scipy, and imports nothing
else from this project. Drop it into any repository that can produce two
aligned participant-by-outcome tables.

This module computes and reports; it does not interpret. No thresholds for
"good" or "adequate" correspondence are encoded, and no published values are
offered for comparison. That is deliberate: what counts as adequate depends on
the use the twins are being put to, and is a decision for the research team
rather than for a library.

Reference
---------
Stojanov, A. (2026). Do participant-matched LLM personas approximate human
survey data? *Personality and Individual Differences*, 261, 113915.
https://doi.org/10.1016/j.paid.2026.113915

-------------------------------------------------------------------------------
WHY THIS METRIC SET
-------------------------------------------------------------------------------

The upstream Twin-2K-500 evaluation reports accuracy as a range-normalized mean
absolute deviation, plus a correlation per outcome. That answers "how close, on
average" but leaves three questions a reviewer will ask unanswered:

1. Is the twin wrong *systematically* (shifted mean) or *noisily* (right on
   average, wrong per person)? Those have opposite implications: a systematic
   shift can be calibrated away, random error cannot.
2. Do twins reproduce between-person **variability**, or do they compress
   everyone toward a central tendency? A set of twins that all answer near the
   sample mean can show respectable average accuracy while carrying no
   individual-differences signal at all.
3. Does a twin reproduce the *shape* of its own participant's profile across
   outcomes -- high on this, low on that -- independent of overall level?

Stojanov's metric set answers all three, and does so with a published
comparison point: across 91 constructs and 177 participants, GPT-4o personas
matched on demographics plus Big Five scores achieved a within-person profile
correlation of r = .19 and a construct-level correlation of r = .21. That gives
any new result an external reference rather than a bare number.

-------------------------------------------------------------------------------
THREE LEVELS OF CORRESPONDENCE
-------------------------------------------------------------------------------

The metrics are organized by what varies and what is averaged over. Keeping
them separate matters, because a method can do well at one level and fail at
another -- which is precisely Stojanov's finding.

**Group level** (one value per outcome, across participants)
    Does the twin sample sit where the human sample sits, and spread how the
    human sample spreads?

    - ``glass_delta``        (M_twin - M_human) / SD_human
    - ``sd_ratio``           SD_twin / SD_human
    - ``dispersion_mismatch`` |log(SD_twin / SD_human)|

**Individual level** (one value per participant, across outcomes)
    Does *this* twin resemble *this* person?

    - ``person_mad``   mean over outcomes of |human - twin| / SD_human
    - ``profile_r``    correlation of a participant's human and twin profiles

**Outcome level** (one value per outcome, across participants)
    Does the twin rank people the way the humans ranked?

    - ``construct_r``  Pearson r between human and twin scores

**Error decomposition** (one value per outcome)
    Is the error systematic or random?

    - ``abs_bias``      |mean(twin - human)| / SD_human
    - ``rmse_std``      RMSE / SD_human
    - ``error_sd_std``  SD(twin - human) / SD_human

-------------------------------------------------------------------------------
WHY EVERYTHING IS SCALED BY THE HUMAN SD
-------------------------------------------------------------------------------

Every standardized quantity here divides by the **human** standard deviation,
never a pooled one. This follows Stojanov's use of Glass's Delta (Lakens 2013)
and is deliberate for two reasons.

First, the human distribution is the reference the twin is trying to reproduce;
the question is "how far off, in human units", so human units should be the
denominator.

Second, and more importantly: **twin variance is itself a quantity being
measured.** Twins are known to under-disperse -- Stojanov reports a median SD
ratio of 0.36 for demographics-only personas, meaning twin scores spanned about
a third of the human spread. A pooled denominator would absorb that compression
into the yardstick, so a method that collapses everyone onto the mean would be
rewarded with a smaller apparent error. Holding the denominator fixed at the
human SD keeps the compression visible, and keeps every metric on a common
scale across outcomes measured on different response formats.

The dispersion mismatch is logged for a related reason, stated in the paper:
on a raw ratio, over- and under-dispersion are asymmetric (0.5 and 2.0 are
equally wrong but sit at different distances from 1). The log makes them
symmetric and comparable across outcomes with different scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "CorrespondenceResult",
    "load_wide_csv",
    "load_long_format",
    "compute_correspondence",
    "compare_arms",
    "fisher_mean",
]

#: Context for reading a correlation, from the sources Stojanov cites.
#: Relative tolerance below which a vector counts as constant. Tiling a mean
#: across rows leaves floating-point noise of order 1e-16 rather than exact
#: zeros, and treating that as genuine spread yields an SD ratio near 1e-16 and
#: a dispersion mismatch around 35 -- a single such outcome would dominate the
#: mean across outcomes. Scaled by the magnitude of the values so it behaves the
#: same on a 1-7 Likert and a 0-100 slider.
_CONSTANT_RTOL = 1e-9


def _is_constant(values: np.ndarray) -> bool:
    """True if a vector has no meaningful spread."""
    values = values[np.isfinite(values)]
    if values.size < 2:
        return True
    scale = max(1.0, float(np.max(np.abs(values))))
    return bool(np.std(values, ddof=1) < _CONSTANT_RTOL * scale)


def _safe_sd(values: np.ndarray, ddof: int = 1, *, zero_ok: bool = False) -> float:
    """Sample SD.

    Args:
        values: Input vector; non-finite entries are dropped.
        ddof: Delta degrees of freedom.
        zero_ok: If True, a genuine zero is returned as 0.0. Use this for the
            SD of *errors*, where zero variability is a correct and meaningful
            answer. Leave False for a denominator, where zero means the metric
            is undefined.

    Returns:
        The SD, 0.0, or NaN.
    """
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    sd = float(np.std(values, ddof=ddof))
    if sd > 0:
        return sd
    return 0.0 if zero_ok else float("nan")


#: Bound applied before arctanh, which is infinite at exactly +/-1. Set tight
#: enough that a correlation of 1.0 round-trips to within 1e-10 -- a looser
#: bound such as 0.9999 would silently report a perfect match as 0.9999, which
#: looks like a bug in the metric rather than a property of the transform.
_FISHER_CLIP = 1.0 - 1e-10


def fisher_mean(correlations: Iterable[float], weights: Iterable[float] | None = None) -> float:
    """Average correlations through Fisher's z transform.

    Correlations are not linear in the quantity they estimate, so averaging
    them arithmetically is biased toward zero. Stojanov applies Fisher's r-to-z
    before comparing conditions; we apply it to aggregation as well.

    One property worth knowing: z grows without bound as r approaches 1, so a
    single near-perfect correlation pulls a Fisher mean upward far more than it
    would pull an arithmetic mean. That is the transform behaving correctly --
    an r of .99 really is far more extreme than an r of .90 -- but it means a
    Fisher mean should not be read without also looking at the distribution of
    the individual correlations, which this module reports per outcome.

    Args:
        correlations: Correlation values; NaNs are dropped.
        weights: Optional weights, typically sample sizes.

    Returns:
        The back-transformed mean correlation, or NaN if none are usable.
    """
    r = np.asarray(list(correlations), dtype=float)
    mask = np.isfinite(r)
    if not mask.any():
        return float("nan")
    r = np.clip(r[mask], -_FISHER_CLIP, _FISHER_CLIP)
    z = np.arctanh(r)
    if weights is not None:
        w = np.asarray(list(weights), dtype=float)[mask]
        if np.nansum(w) <= 0:
            return float("nan")
        return float(np.tanh(np.nansum(z * w) / np.nansum(w)))
    return float(np.tanh(np.nanmean(z)))


@dataclass
class CorrespondenceResult:
    """Correspondence between one twin arm and the human data.

    Attributes:
        outcome_level: One row per outcome, with group-level, outcome-level and
            error metrics.
        participant_level: One row per participant, with person MAD and
            profile r.
        summary: Scalar summaries across outcomes and participants.
        n_outcomes: Outcomes with usable data.
        n_participants: Participants with usable data.
        arm: Label for this arm.
        warnings: Conditions that make a metric hard to interpret.
    """

    outcome_level: pd.DataFrame
    participant_level: pd.DataFrame
    summary: dict[str, float]
    n_outcomes: int
    n_participants: int
    arm: str = "twin"
    warnings: list[str] = field(default_factory=list)

    def report(self) -> str:
        """Render a readable summary of the metrics.

        Reports values only. Whether a given level of correspondence is
        adequate depends on what the twins are being used for, and is left to
        the reader.

        Returns:
            A formatted multi-line report.
        """
        rows = [
            ("Group level", None),
            ("  mean |Glass's delta|", "abs_glass_delta"),
            ("  dispersion mismatch", "dispersion_mismatch"),
            ("  median SD ratio", "sd_ratio_median"),
            ("Individual level", None),
            ("  person-level MAD", "person_mad"),
            ("  within-person profile r", "profile_r"),
            ("Outcome level", None),
            ("  outcome-level r (Fisher)", "construct_r"),
            ("Error decomposition", None),
            ("  standardized |bias|", "abs_bias"),
            ("  standardized RMSE", "rmse_std"),
            ("  standardized error SD", "error_sd_std"),
        ]
        header = f"Correspondence: {self.arm}  "
        header += f"(k = {self.n_outcomes} outcomes, n = {self.n_participants} participants)"
        lines = [header, "=" * len(header)]
        for label, key in rows:
            if key is None:
                lines.append(label)
                continue
            lines.append(f"{label:<34}{self.summary.get(key, float('nan')):>12.3f}")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def _align(
    human: pd.DataFrame,
    twin: pd.DataFrame,
    id_column: str,
    outcomes: Sequence[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Reduce both frames to shared participants and shared numeric outcomes."""
    warnings: list[str] = []

    for name, frame in (("human", human), ("twin", twin)):
        if id_column not in frame.columns:
            raise KeyError(
                f"{name} data has no column {id_column!r}. Columns seen: "
                f"{list(frame.columns)[:10]}"
            )

    h = human.copy()
    t = twin.copy()
    h[id_column] = h[id_column].astype(str).str.strip()
    t[id_column] = t[id_column].astype(str).str.strip()

    for name, frame in (("human", h), ("twin", t)):
        dupes = frame[id_column].duplicated().sum()
        if dupes:
            warnings.append(
                f"{dupes} duplicate participant id(s) in the {name} data; "
                "keeping the first of each."
            )
    h = h.drop_duplicates(subset=[id_column]).set_index(id_column)
    t = t.drop_duplicates(subset=[id_column]).set_index(id_column)

    shared_ids = sorted(set(h.index) & set(t.index))
    if not shared_ids:
        raise ValueError(
            "No participant ids are present in both the human and twin data. "
            "Each twin must be matched to the human it was built from; without "
            "that pairing none of these metrics is defined."
        )
    dropped = (len(h) - len(shared_ids)) + (len(t) - len(shared_ids))
    if dropped:
        warnings.append(
            f"{len(shared_ids)} participants matched; {dropped} unmatched "
            "row(s) dropped."
        )

    if outcomes is None:
        candidates = [c for c in h.columns if c in t.columns]
    else:
        missing = [c for c in outcomes if c not in h.columns or c not in t.columns]
        if missing:
            warnings.append(
                f"{len(missing)} requested outcome(s) absent from one side and "
                f"skipped: {missing[:5]}"
            )
        candidates = [c for c in outcomes if c in h.columns and c in t.columns]

    h = h.loc[shared_ids]
    t = t.loc[shared_ids]

    usable: list[str] = []
    for column in candidates:
        hv = pd.to_numeric(h[column], errors="coerce")
        tv = pd.to_numeric(t[column], errors="coerce")
        both = hv.notna() & tv.notna()
        if both.sum() < 3:
            continue
        if not np.isfinite(_safe_sd(hv[both].to_numpy())):
            warnings.append(
                f"outcome {column!r} has no variance among humans; every "
                "standardized metric would be undefined, so it is excluded."
            )
            continue
        h[column] = hv
        t[column] = tv
        usable.append(column)

    if not usable:
        raise ValueError(
            "No usable outcomes. Each outcome needs numeric values for at "
            "least 3 matched participants and non-zero variance among humans."
        )
    return h[usable], t[usable], usable, warnings


def compute_correspondence(
    human: pd.DataFrame,
    twin: pd.DataFrame,
    *,
    id_column: str = "TWIN_ID",
    outcomes: Sequence[str] | None = None,
    arm: str = "twin",
) -> CorrespondenceResult:
    """Compute the full correspondence metric set for one arm.

    Args:
        human: Wide frame, one row per participant, one column per outcome.
        twin: Same shape, same participant ids and outcome names.
        id_column: Participant identifier present in both frames.
        outcomes: Outcome columns to use. None uses every shared numeric column
            with usable data -- convenient, but naming them explicitly is safer,
            since an unnoticed metadata column would otherwise be scored as an
            outcome.
        arm: Label for this arm, used in reports.

    Returns:
        A :class:`CorrespondenceResult`.

    Raises:
        KeyError: If ``id_column`` is missing.
        ValueError: If no participants match, or no outcome is usable.
    """
    h, t, used, warnings = _align(human, twin, id_column, outcomes)

    # ---- per-outcome metrics ------------------------------------------
    rows: list[dict[str, Any]] = []
    for column in used:
        hv = h[column].to_numpy(dtype=float)
        tv = t[column].to_numpy(dtype=float)
        mask = np.isfinite(hv) & np.isfinite(tv)
        hv, tv = hv[mask], tv[mask]
        n = hv.size

        sd_h = _safe_sd(hv)
        twin_constant = _is_constant(tv)
        sd_t = 0.0 if twin_constant else _safe_sd(tv)
        error = tv - hv

        with np.errstate(invalid="ignore", divide="ignore"):
            glass_delta = (tv.mean() - hv.mean()) / sd_h
            sd_ratio = sd_t / sd_h if np.isfinite(sd_t) else float("nan")
            # A twin vector with no spread has infinite log-ratio. Recorded as
            # undefined and counted separately rather than folded into the mean
            # as a very large number, which would make one collapsed outcome
            # swamp every other.
            if np.isfinite(sd_ratio) and sd_ratio > 0:
                dispersion = abs(float(np.log(sd_ratio)))
            else:
                dispersion = float("nan")
            abs_bias = abs(error.mean()) / sd_h
            rmse_std = float(np.sqrt(np.mean(error**2))) / sd_h
            error_sd_std = _safe_sd(error, zero_ok=True) / sd_h

        if n < 3:
            r, p = float("nan"), float("nan")
        elif twin_constant:
            # Every twin gave the same answer, so the twins impose no ordering
            # on participants. The correlation is genuinely zero information,
            # not missing data: recording 0 keeps the outcome in the average,
            # where NaN would drop it and flatter the result.
            r, p = 0.0, float("nan")
        else:
            r, p = stats.pearsonr(hv, tv)

        rows.append(
            {
                "outcome": column,
                "n": n,
                "mean_human": float(hv.mean()),
                "mean_twin": float(tv.mean()),
                "sd_human": sd_h,
                "sd_twin": float(sd_t),
                "twin_constant": bool(twin_constant),
                "glass_delta": float(glass_delta),
                "abs_glass_delta": float(abs(glass_delta)),
                "sd_ratio": float(sd_ratio),
                "dispersion_mismatch": float(dispersion),
                "construct_r": float(r),
                "construct_r_p": float(p),
                "abs_bias": float(abs_bias),
                "rmse_std": float(rmse_std),
                "error_sd_std": float(error_sd_std),
            }
        )
    outcome_df = pd.DataFrame(rows)

    # ---- per-participant metrics ---------------------------------------
    # Both sides are z-scored against the HUMAN mean and SD of each outcome, as
    # in Stojanov §2.2.2, so that differences in level and spread are removed
    # before profiles are compared. What remains is the shape of the profile.
    sd_map = outcome_df.set_index("outcome")["sd_human"]
    mean_map = outcome_df.set_index("outcome")["mean_human"]
    hz = (h[used] - mean_map) / sd_map
    tz = (t[used] - mean_map) / sd_map

    participant_rows: list[dict[str, Any]] = []
    for pid in h.index:
        hp = hz.loc[pid].to_numpy(dtype=float)
        tp = tz.loc[pid].to_numpy(dtype=float)
        mask = np.isfinite(hp) & np.isfinite(tp)
        k = int(mask.sum())
        mad = float(np.mean(np.abs(hp[mask] - tp[mask]))) if k else float("nan")

        if k < 3:
            profile_r = float("nan")
        elif _is_constant(hp[mask]) or _is_constant(tp[mask]):
            # A flat profile on either side has no shape to match. Zero is the
            # honest value: the twin reproduced no within-person pattern.
            profile_r = 0.0
        else:
            profile_r = float(stats.pearsonr(hp[mask], tp[mask])[0])

        participant_rows.append(
            {
                id_column: pid,
                "n_outcomes": k,
                "person_mad": mad,
                "profile_r": profile_r,
            }
        )
    participant_df = pd.DataFrame(participant_rows)

    # ---- summaries -------------------------------------------------------
    summary = {
        "abs_glass_delta": float(outcome_df["abs_glass_delta"].mean()),
        "mean_glass_delta": float(outcome_df["glass_delta"].mean()),
        "dispersion_mismatch": float(outcome_df["dispersion_mismatch"].mean()),
        "sd_ratio_median": float(outcome_df["sd_ratio"].median()),
        "person_mad": float(participant_df["person_mad"].mean()),
        "profile_r": fisher_mean(participant_df["profile_r"]),
        "construct_r": fisher_mean(outcome_df["construct_r"], outcome_df["n"]),
        "abs_bias": float(outcome_df["abs_bias"].mean()),
        "rmse_std": float(outcome_df["rmse_std"].mean()),
        "error_sd_std": float(outcome_df["error_sd_std"].mean()),
    }

    k_outcomes = len(used)
    n_participants = len(participant_df)

    if k_outcomes < 10:
        warnings.append(
            f"Only {k_outcomes} outcomes. The within-person profile "
            "correlation is computed across outcomes, so with few of them it "
            "is very unstable; Stojanov used 91. Treat profile_r as indicative "
            "at best, and consider reporting person_mad instead."
        )
    if n_participants < 30:
        warnings.append(
            f"Only {n_participants} matched participants. Outcome-level "
            "correlations are computed across participants and will be noisy."
        )
    collapsed = int(outcome_df["twin_constant"].sum())
    if collapsed:
        warnings.append(
            f"{collapsed} of {k_outcomes} outcomes had every twin giving the "
            "same answer. Their dispersion mismatch is undefined (excluded "
            "from that mean) and their correlation is recorded as 0. This is "
            "the most severe form of the variance compression Stojanov "
            "reports, and is worth naming explicitly in a write-up."
        )

    low_var = int((outcome_df["sd_ratio"] < 0.5).sum())
    if low_var:
        warnings.append(
            f"{low_var} of {k_outcomes} outcomes have twin SD below half the "
            "human SD -- substantial variance compression. Expect correlations "
            "to be attenuated for mechanical reasons as well as substantive "
            "ones."
        )

    return CorrespondenceResult(
        outcome_level=outcome_df,
        participant_level=participant_df,
        summary=summary,
        n_outcomes=k_outcomes,
        n_participants=n_participants,
        arm=arm,
        warnings=warnings,
    )


def _cohens_dz(differences: np.ndarray) -> float:
    """Cohen's d_z for paired differences: mean(diff) / SD(diff)."""
    differences = differences[np.isfinite(differences)]
    if differences.size < 2:
        return float("nan")
    sd = _safe_sd(differences)
    return float(np.mean(differences) / sd) if np.isfinite(sd) else float("nan")


def compare_arms(
    arm_a: CorrespondenceResult,
    arm_b: CorrespondenceResult,
    *,
    id_column: str = "TWIN_ID",
) -> pd.DataFrame:
    """Compare two arms on each metric with paired tests.

    Follows Stojanov §2.2.5: paired-samples t-tests across outcomes or across
    participants as appropriate, Wilcoxon signed-rank as a non-parametric
    check, and Cohen's d_z as the effect size.

    The comparison is paired because the two arms describe the *same* outcomes
    and the *same* participants, differing only in how the twin was built. That
    pairing is what makes the contrast informative: it holds the instrument and
    the sample fixed.

    Args:
        arm_a: Baseline arm, e.g. the no-persona control.
        arm_b: Comparison arm, e.g. the full-persona twin.
        id_column: Participant identifier in the participant-level frames.

    Returns:
        One row per metric: both means, the difference, t and p, Wilcoxon p,
        Cohen's d_z, and the unit of pairing.
    """
    outcome_metrics = [
        "abs_glass_delta",
        "dispersion_mismatch",
        "sd_ratio",
        "construct_r",
        "abs_bias",
        "rmse_std",
        "error_sd_std",
    ]
    participant_metrics = ["person_mad", "profile_r"]
    rows: list[dict[str, Any]] = []

    def _add(metric: str, a: pd.Series, b: pd.Series, unit: str) -> None:
        paired = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        if len(paired) < 3:
            return
        av = paired["a"].to_numpy(dtype=float)
        bv = paired["b"].to_numpy(dtype=float)
        # Fisher-transform correlations before testing, per Stojanov §2.2.3.
        if metric in {"construct_r", "profile_r"}:
            av = np.arctanh(np.clip(av, -_FISHER_CLIP, _FISHER_CLIP))
            bv = np.arctanh(np.clip(bv, -_FISHER_CLIP, _FISHER_CLIP))
        diff = bv - av
        try:
            t_stat, t_p = stats.ttest_rel(bv, av)
        except Exception:  # noqa: BLE001
            t_stat, t_p = float("nan"), float("nan")
        try:
            _, w_p = stats.wilcoxon(bv, av)
        except Exception:  # noqa: BLE001
            w_p = float("nan")
        rows.append(
            {
                "metric": metric,
                "unit": unit,
                "n_pairs": len(paired),
                f"mean_{arm_a.arm}": float(paired["a"].mean()),
                f"mean_{arm_b.arm}": float(paired["b"].mean()),
                "difference": float(paired["b"].mean() - paired["a"].mean()),
                "t": float(t_stat),
                "p_ttest": float(t_p),
                "p_wilcoxon": float(w_p),
                "cohens_dz": _cohens_dz(diff),
            }
        )

    a_out = arm_a.outcome_level.set_index("outcome")
    b_out = arm_b.outcome_level.set_index("outcome")
    for metric in outcome_metrics:
        if metric in a_out.columns and metric in b_out.columns:
            _add(metric, a_out[metric], b_out[metric], "outcome")

    a_par = arm_a.participant_level.set_index(id_column)
    b_par = arm_b.participant_level.set_index(id_column)
    for metric in participant_metrics:
        if metric in a_par.columns and metric in b_par.columns:
            _add(metric, a_par[metric], b_par[metric], "participant")

    return pd.DataFrame(rows)




def load_wide_csv(
    path: str | Path,
    *,
    id_column: str = "TWIN_ID",
    skip_header_rows: Sequence[int] | None = (1, 2),
) -> pd.DataFrame:
    """Read a consolidated CSV written by :mod:`kli.responses_to_csv`.

    Those files carry the Qualtrics convention of extra header rows beneath the
    column names (question text, then metadata), which must be skipped or every
    column is read as text.

    Args:
        path: Path to the CSV.
        id_column: Participant identifier column.
        skip_header_rows: Row indices to drop after the header. Pass None for a
            plain single-header CSV.

    Returns:
        A wide frame indexed by position, with ``id_column`` as a column.

    Raises:
        KeyError: If the identifier column is absent.
    """
    frame = pd.read_csv(
        path,
        skiprows=list(skip_header_rows) if skip_header_rows else None,
        low_memory=False,
    )
    if id_column not in frame.columns:
        raise KeyError(
            f"{path} has no {id_column!r} column. Columns: "
            f"{list(frame.columns)[:10]}"
        )
    return frame


def load_long_format(
    path: str | Path,
    *,
    id_column: str = "TWIN_ID",
    variable_column: str = "variable_name",
    value_column: str = "value",
    type_column: str = "respondent_type",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the long-format file the evaluation layer produces, and split it.

    ``meta analysis individual level.csv`` holds human and twin rows stacked,
    which is convenient to produce and useless to analyse directly. This pivots
    it into the two aligned wide frames the metrics need.

    Args:
        path: Path to the long-format CSV (``.csv`` or ``.csv.gz``).
        id_column: Participant identifier.
        variable_column: Column naming the outcome.
        value_column: Column holding the value.
        type_column: Column distinguishing human from twin rows.

    Returns:
        ``(human_wide, twin_wide)``.

    Raises:
        ValueError: If either respondent type is absent.
    """
    long = pd.read_csv(path, low_memory=False)
    missing = {id_column, variable_column, value_column, type_column} - set(long.columns)
    if missing:
        raise ValueError(f"{path} is missing expected column(s): {sorted(missing)}")

    present = set(long[type_column].astype(str).str.lower().unique())
    for wanted in ("human", "twin"):
        if wanted not in present:
            raise ValueError(
                f"{path} contains no rows with {type_column} == {wanted!r}. "
                f"Values present: {sorted(present)}"
            )

    def _pivot(kind: str) -> pd.DataFrame:
        subset = long[long[type_column].astype(str).str.lower() == kind]
        wide = subset.pivot_table(
            index=id_column,
            columns=variable_column,
            values=value_column,
            aggfunc="first",
        ).reset_index()
        wide.columns.name = None
        return wide

    return _pivot("human"), _pivot("twin")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Examples::

        # one arm, from the two consolidated CSVs
        python -m kli.correspondence \\
            --human  runs/my_study/human/consolidated_original_answers_values.csv \\
            --twin   runs/my_study/csv_output/consolidated_llm_values.csv

        # compare the persona arm against the no-persona control
        python -m kli.correspondence \\
            --human   .../human.csv \\
            --twin    .../twin_full_persona.csv \\
            --control .../twin_empty_persona.csv \\
            --out     runs/my_study/correspondence
    """
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(
        description="Human-twin correspondence metrics (Stojanov 2026)."
    )
    parser.add_argument("--human", help="Human wide CSV.")
    parser.add_argument("--twin", help="Twin wide CSV (the main arm).")
    parser.add_argument(
        "--control",
        default=None,
        help="Optional second twin arm to compare against, e.g. the "
        "no-persona or scrambled-persona control.",
    )
    parser.add_argument(
        "--long-format",
        default=None,
        help="Alternative to --human/--twin: a 'meta analysis individual "
        "level.csv' holding both.",
    )
    parser.add_argument("--id-column", default="TWIN_ID")
    parser.add_argument(
        "--outcomes",
        default=None,
        help="Comma-separated outcome columns. Strongly recommended: without "
        "it every shared numeric column is treated as an outcome.",
    )
    parser.add_argument(
        "--no-skip-header-rows",
        action="store_true",
        help="Input CSVs have a single header row (not the Qualtrics layout).",
    )
    parser.add_argument("--out", default=None, help="Directory for result tables.")
    args = parser.parse_args(argv)

    skip = None if args.no_skip_header_rows else (1, 2)
    outcomes = [c.strip() for c in args.outcomes.split(",")] if args.outcomes else None

    if args.long_format:
        human, twin = load_long_format(args.long_format, id_column=args.id_column)
        control = None
    else:
        if not args.human or not args.twin:
            parser.error("provide --human and --twin, or --long-format")
        human = load_wide_csv(args.human, id_column=args.id_column, skip_header_rows=skip)
        twin = load_wide_csv(args.twin, id_column=args.id_column, skip_header_rows=skip)
        control = (
            load_wide_csv(args.control, id_column=args.id_column, skip_header_rows=skip)
            if args.control
            else None
        )

    main_arm = compute_correspondence(
        human, twin, id_column=args.id_column, outcomes=outcomes, arm="twin"
    )
    print(main_arm.report())

    comparison = None
    if control is not None:
        control_arm = compute_correspondence(
            human, control, id_column=args.id_column, outcomes=outcomes, arm="control"
        )
        print()
        print(control_arm.report())
        comparison = compare_arms(control_arm, main_arm, id_column=args.id_column)
        print()
        print("Paired comparison (twin vs control):")
        print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        main_arm.outcome_level.to_csv(out / "correspondence_by_outcome.csv", index=False)
        main_arm.participant_level.to_csv(
            out / "correspondence_by_participant.csv", index=False
        )
        (out / "correspondence_summary.json").write_text(
            _json.dumps(
                {
                    "arm": main_arm.arm,
                    "n_outcomes": main_arm.n_outcomes,
                    "n_participants": main_arm.n_participants,
                    "summary": main_arm.summary,
                    "warnings": main_arm.warnings,
                    "reference": "Stojanov (2026), Personality and Individual "
                    "Differences 261, 113915",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if comparison is not None:
            comparison.to_csv(out / "arm_comparison.csv", index=False)
        print(f"\nwritten to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
