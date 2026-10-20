"""Tests for the correspondence metrics (Stojanov 2026).

ORIGINAL WORK (Kravis Leadership Institute).

Each metric is checked against a case whose value can be derived by hand, so a
failure points at a specific formula rather than at "the numbers moved". The
constructed cases also double as documentation of what each metric means:

    perfect      twin == human                  every metric at its ideal
    shifted      twin == human + 1 human SD     pure systematic error
    compressed   spread halved, order kept      pure variance compression
    flat         every twin gives the same      total compression
    noise        twin unrelated to human        no signal

Run with:  python -m pytest tests/test_correspondence.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kli.correspondence import (
    STOJANOV_2026_BENCHMARKS,
    compare_arms,
    compute_correspondence,
    fisher_mean,
    interpret_correlation,
)

N_PARTICIPANTS = 120
N_OUTCOMES = 20


@pytest.fixture
def human() -> pd.DataFrame:
    """A human dataset with known distributional properties."""
    rng = np.random.default_rng(20260520)
    frame = pd.DataFrame(
        rng.normal(4.0, 1.2, size=(N_PARTICIPANTS, N_OUTCOMES)),
        columns=[f"dv{j}" for j in range(N_OUTCOMES)],
    )
    frame.insert(0, "TWIN_ID", [f"p{i}" for i in range(N_PARTICIPANTS)])
    return frame


def _derive(human: pd.DataFrame, fn) -> pd.DataFrame:
    outcomes = [c for c in human.columns if c != "TWIN_ID"]
    twin = human.copy()
    twin[outcomes] = fn(human[outcomes])
    return twin


@pytest.fixture
def perfect(human: pd.DataFrame) -> pd.DataFrame:
    return _derive(human, lambda d: d)


@pytest.fixture
def shifted(human: pd.DataFrame) -> pd.DataFrame:
    """Every value raised by exactly one human SD."""
    return _derive(human, lambda d: d + d.std(ddof=1))


@pytest.fixture
def compressed(human: pd.DataFrame) -> pd.DataFrame:
    """Spread halved around the mean; participant ordering untouched."""
    return _derive(human, lambda d: d.mean() + 0.5 * (d - d.mean()))


@pytest.fixture
def flat(human: pd.DataFrame) -> pd.DataFrame:
    """Every twin gives the human mean: total variance collapse."""
    return _derive(
        human,
        lambda d: pd.DataFrame(
            np.tile(d.mean().to_numpy(), (len(d), 1)), columns=d.columns, index=d.index
        ),
    )


@pytest.fixture
def noise(human: pd.DataFrame) -> pd.DataFrame:
    """Same distribution, no relationship to the individual."""
    rng = np.random.default_rng(7)
    return _derive(
        human,
        lambda d: pd.DataFrame(
            rng.normal(4.0, 1.2, size=d.shape), columns=d.columns, index=d.index
        ),
    )


class TestPerfectCorrespondence:
    """A twin identical to its human should score ideally on everything."""

    def test_all_metrics_at_ideal(self, human, perfect) -> None:
        s = compute_correspondence(human, perfect).summary
        assert s["abs_glass_delta"] == pytest.approx(0.0, abs=1e-9)
        assert s["sd_ratio_median"] == pytest.approx(1.0, abs=1e-9)
        assert s["dispersion_mismatch"] == pytest.approx(0.0, abs=1e-9)
        assert s["person_mad"] == pytest.approx(0.0, abs=1e-9)
        assert s["profile_r"] == pytest.approx(1.0, abs=1e-6)
        assert s["construct_r"] == pytest.approx(1.0, abs=1e-6)
        assert s["rmse_std"] == pytest.approx(0.0, abs=1e-9)

    def test_zero_error_sd_is_zero_not_missing(self, human, perfect) -> None:
        """Zero error variability is a real answer, not undefined."""
        s = compute_correspondence(human, perfect).summary
        assert s["error_sd_std"] == pytest.approx(0.0, abs=1e-9)
        assert np.isfinite(s["error_sd_std"])


class TestSystematicShift:
    """A constant offset is pure bias: detectable, and in principle correctable."""

    def test_glass_delta_recovers_the_shift(self, human, shifted) -> None:
        s = compute_correspondence(human, shifted).summary
        assert s["abs_glass_delta"] == pytest.approx(1.0, abs=1e-6)
        assert s["mean_glass_delta"] == pytest.approx(1.0, abs=1e-6)

    def test_bias_absorbs_all_error(self, human, shifted) -> None:
        """Bias equals RMSE and error SD is zero: the error is entirely systematic."""
        s = compute_correspondence(human, shifted).summary
        assert s["abs_bias"] == pytest.approx(1.0, abs=1e-6)
        assert s["rmse_std"] == pytest.approx(1.0, abs=1e-6)
        assert s["error_sd_std"] == pytest.approx(0.0, abs=1e-9)

    def test_shift_does_not_disturb_rank_or_spread(self, human, shifted) -> None:
        s = compute_correspondence(human, shifted).summary
        assert s["construct_r"] == pytest.approx(1.0, abs=1e-6)
        assert s["sd_ratio_median"] == pytest.approx(1.0, abs=1e-6)


class TestVarianceCompression:
    """The failure mode raw accuracy hides and these metrics expose."""

    def test_sd_ratio_recovers_the_compression(self, human, compressed) -> None:
        s = compute_correspondence(human, compressed).summary
        assert s["sd_ratio_median"] == pytest.approx(0.5, abs=1e-6)

    def test_dispersion_mismatch_is_log_of_the_ratio(self, human, compressed) -> None:
        s = compute_correspondence(human, compressed).summary
        assert s["dispersion_mismatch"] == pytest.approx(abs(np.log(0.5)), abs=1e-6)

    def test_compression_is_invisible_to_correlation_and_bias(
        self, human, compressed
    ) -> None:
        """Why the dispersion metrics are needed at all.

        Halving everyone's spread leaves the correlation perfect and the mean
        unchanged. A study reporting only r and mean accuracy would call this
        a success.
        """
        s = compute_correspondence(human, compressed).summary
        assert s["construct_r"] == pytest.approx(1.0, abs=1e-6)
        assert s["abs_glass_delta"] == pytest.approx(0.0, abs=1e-6)
        assert s["abs_bias"] == pytest.approx(0.0, abs=1e-6)
        assert s["dispersion_mismatch"] > 0.6  # but this catches it

    def test_over_and_under_dispersion_are_symmetric(self, human) -> None:
        """The reason the ratio is logged."""
        half = _derive(human, lambda d: d.mean() + 0.5 * (d - d.mean()))
        double = _derive(human, lambda d: d.mean() + 2.0 * (d - d.mean()))
        a = compute_correspondence(human, half).summary["dispersion_mismatch"]
        b = compute_correspondence(human, double).summary["dispersion_mismatch"]
        assert a == pytest.approx(b, abs=1e-6)


class TestTotalCollapse:
    """Every twin answering identically."""

    def test_sd_ratio_is_zero(self, human, flat) -> None:
        assert compute_correspondence(human, flat).summary["sd_ratio_median"] == 0.0

    def test_correlation_is_zero_not_missing(self, human, flat) -> None:
        """Recording 0 keeps the outcome in the average; NaN would flatter it."""
        s = compute_correspondence(human, flat).summary
        assert s["construct_r"] == pytest.approx(0.0, abs=1e-9)
        assert s["profile_r"] == pytest.approx(0.0, abs=1e-9)

    def test_dispersion_is_undefined_not_enormous(self, human, flat) -> None:
        """log(0) is unbounded; one collapsed outcome must not swamp the mean."""
        result = compute_correspondence(human, flat)
        assert not np.isfinite(result.summary["dispersion_mismatch"])
        assert result.outcome_level["twin_constant"].all()

    def test_collapse_is_flagged(self, human, flat) -> None:
        result = compute_correspondence(human, flat)
        assert any("same answer" in w for w in result.warnings)


class TestNoSignal:
    """An unrelated twin drawn from the right distribution."""

    def test_correlations_near_zero(self, human, noise) -> None:
        s = compute_correspondence(human, noise).summary
        assert abs(s["construct_r"]) < 0.15
        assert abs(s["profile_r"]) < 0.15

    def test_rmse_approaches_root_two(self, human, noise) -> None:
        """Two independent draws of the same SD differ by sqrt(2) SDs."""
        s = compute_correspondence(human, noise).summary
        assert s["rmse_std"] == pytest.approx(np.sqrt(2), abs=0.15)

    def test_spread_looks_right_despite_no_signal(self, human, noise) -> None:
        """Distributional realism is not individual-level accuracy."""
        s = compute_correspondence(human, noise).summary
        assert s["sd_ratio_median"] == pytest.approx(1.0, abs=0.2)
        assert abs(s["construct_r"]) < 0.15


class TestStandardizationChoice:
    """Everything is scaled by the human SD, never a pooled SD."""

    def test_compression_is_not_rewarded(self, human) -> None:
        """A pooled denominator would shrink with the twin's own variance.

        Scaling by the human SD keeps the yardstick fixed, so collapsing
        everyone toward the mean cannot buy a smaller apparent error.
        """
        mild = _derive(human, lambda d: d.mean() + 0.5 * (d - d.mean()))
        severe = _derive(human, lambda d: d.mean() + 0.1 * (d - d.mean()))
        mild_rmse = compute_correspondence(human, mild).summary["rmse_std"]
        severe_rmse = compute_correspondence(human, severe).summary["rmse_std"]
        assert severe_rmse > mild_rmse

    def test_denominator_is_the_human_sd(self, human, shifted) -> None:
        """A shift of one human SD scores exactly 1.0, by construction."""
        row = compute_correspondence(human, shifted).outcome_level.iloc[0]
        assert row["glass_delta"] == pytest.approx(1.0, abs=1e-6)


class TestAlignment:
    def test_requires_matched_participants(self, human, perfect) -> None:
        relabelled = perfect.copy()
        relabelled["TWIN_ID"] = [f"other{i}" for i in range(len(relabelled))]
        with pytest.raises(ValueError, match="present in both"):
            compute_correspondence(human, relabelled)

    def test_uses_only_shared_participants(self, human, perfect) -> None:
        result = compute_correspondence(human.iloc[:50], perfect)
        assert result.n_participants == 50
        assert any("unmatched" in w for w in result.warnings)

    def test_missing_id_column_raises(self, human, perfect) -> None:
        with pytest.raises(KeyError, match="MISSING"):
            compute_correspondence(human, perfect, id_column="MISSING")

    def test_constant_human_outcome_is_excluded(self, human, perfect) -> None:
        """Zero human SD makes every standardized metric undefined."""
        h = human.copy()
        t = perfect.copy()
        h["dv0"] = 3.0
        t["dv0"] = 3.0
        result = compute_correspondence(h, t)
        assert "dv0" not in set(result.outcome_level["outcome"])
        assert any("no variance among humans" in w for w in result.warnings)

    def test_explicit_outcome_selection(self, human, perfect) -> None:
        result = compute_correspondence(human, perfect, outcomes=["dv0", "dv1"])
        assert result.n_outcomes == 2


class TestWarnings:
    def test_warns_on_too_few_outcomes_for_profile_r(self, human, perfect) -> None:
        """Profile r across 3 outcomes is not interpretable; Stojanov used 91."""
        result = compute_correspondence(human, perfect, outcomes=["dv0", "dv1", "dv2"])
        assert any("profile_r" in w for w in result.warnings)

    def test_warns_on_variance_compression(self, human, compressed) -> None:
        result = compute_correspondence(human, compressed)
        assert any("compression" in w for w in result.warnings)


class TestArmComparison:
    def test_detects_a_better_arm(self, human, compressed, noise) -> None:
        """A compressed-but-ordered arm should beat an unrelated one."""
        good = compute_correspondence(human, compressed, arm="persona")
        bad = compute_correspondence(human, noise, arm="control")
        table = compare_arms(bad, good).set_index("metric")
        assert table.loc["construct_r", "difference"] > 0
        assert table.loc["construct_r", "p_ttest"] < 0.05

    def test_reports_both_tests_and_an_effect_size(self, human, compressed, noise) -> None:
        table = compare_arms(
            compute_correspondence(human, noise, arm="control"),
            compute_correspondence(human, compressed, arm="persona"),
        )
        for column in ("t", "p_ttest", "p_wilcoxon", "cohens_dz", "n_pairs", "unit"):
            assert column in table.columns

    def test_pairs_outcome_and_participant_metrics_separately(
        self, human, compressed, noise
    ) -> None:
        table = compare_arms(
            compute_correspondence(human, noise, arm="control"),
            compute_correspondence(human, compressed, arm="persona"),
        ).set_index("metric")
        assert table.loc["construct_r", "unit"] == "outcome"
        assert table.loc["person_mad", "unit"] == "participant"
        assert table.loc["person_mad", "n_pairs"] == N_PARTICIPANTS

    def test_identical_arms_show_no_difference(self, human, compressed) -> None:
        a = compute_correspondence(human, compressed, arm="a")
        b = compute_correspondence(human, compressed, arm="b")
        table = compare_arms(a, b)
        assert np.allclose(table["difference"].to_numpy(), 0.0, atol=1e-9)


class TestFisherMean:
    def test_exceeds_arithmetic_mean_for_spread_correlations(self) -> None:
        """Arithmetic averaging of correlations is biased toward zero."""
        values = [0.1, 0.5, 0.9]
        assert fisher_mean(values) > float(np.mean(values))

    def test_handles_extremes_without_overflow(self) -> None:
        assert np.isfinite(fisher_mean([1.0, -1.0, 0.5]))

    def test_ignores_missing(self) -> None:
        assert fisher_mean([0.5, float("nan")]) == pytest.approx(0.5, abs=1e-9)

    def test_all_missing_gives_missing(self) -> None:
        assert not np.isfinite(fisher_mean([float("nan")]))


class TestBenchmarksAndInterpretation:
    def test_published_arms_are_present(self) -> None:
        for arm in (
            "demographics_only",
            "demographics_plus_big5",
            "random_big5_control",
        ):
            assert arm in STOJANOV_2026_BENCHMARKS

    def test_benchmark_values_match_the_paper(self) -> None:
        """Table 2 of Stojanov (2026), spot-checked."""
        big5 = STOJANOV_2026_BENCHMARKS["demographics_plus_big5"]
        assert big5["profile_r"] == pytest.approx(0.187)
        assert big5["construct_r"] == pytest.approx(0.213)
        assert big5["sd_ratio_median"] == pytest.approx(0.613)

    def test_interpretation_flags_weak_correlations(self) -> None:
        assert "below" in interpret_correlation(0.05)

    def test_interpretation_mentions_the_reliability_floor(self) -> None:
        assert "0.70" in interpret_correlation(0.21)
