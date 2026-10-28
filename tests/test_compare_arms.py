"""Tests for the arm comparison behind H1-H3, and the shipped arm configs.

ORIGINAL WORK (Kravis Leadership Institute).

Run with:  python -m pytest tests/test_compare_arms.py -v
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from kli.compare_arms import compare_arms
from kli.config import load_config
from kli.run_simulation import run
from kli.survey_tree import walk_questions
from test_sjt_arm import PIDS, _config, study  # noqa: F401 - fixture

REPO = Path(__file__).resolve().parents[1]
ARMS = {
    "matched_off": (False, False, False),
    "matched_on": (False, False, True),
    "scrambled_off": (False, True, False),
    "scrambled_on": (False, True, True),
    "empty_off": (True, False, False),
    "empty_on": (True, False, True),
}


class TestShippedConfigs:
    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_arm_config_sets_exactly_its_flags(self, name: str) -> None:
        config = load_config(REPO / "configs" / "study" / "arms" / f"{name}.yaml")
        sim = config["text_simulation"]
        combine = sim["create_text_simulation_input"]
        empty, scramble, include = ARMS[name]
        assert combine["empty_persona_prompt"] is empty
        assert combine["scramble_personas"] is scramble
        assert sim["sjt_split"]["include_training_sjts_in_persona"] is include
        # Each arm writes only under its own runs/ directory.
        assert f"runs/study/{name}/" in sim["LLM_simulation"]["output_updated_questions_dir"]
        assert f"runs/study/{name}/" in combine["output_combined_prompts_dir"]

    def test_arms_differ_only_in_arm_lines(self) -> None:
        base = load_config(REPO / "configs" / "study" / "arms" / "matched_off.yaml")
        for name in ARMS:
            other = load_config(REPO / "configs" / "study" / "arms" / f"{name}.yaml")
            for config in (base, other):
                sim = config["text_simulation"]
                sim["create_text_simulation_input"].pop("empty_persona_prompt")
                sim["create_text_simulation_input"].pop("scramble_personas")
                sim["sjt_split"].pop("include_training_sjts_in_persona")
            text = json.dumps(other, sort_keys=True).replace(name, "ARM")
            assert text == json.dumps(base, sort_keys=True).replace("matched_off", "ARM")
            base = load_config(REPO / "configs" / "study" / "arms" / "matched_off.yaml")

    def test_comparison_config_names_the_three_hypotheses(self) -> None:
        block = load_config(REPO / "configs" / "study" / "3_compare_arms.yaml")["compare_arms"]
        assert set(block["contrasts"]) == {"H1", "H2", "H3"}
        for pair in block["contrasts"].values():
            assert pair["focal"] == "meso"
            assert pair["comparator"] in block["arms"]
        assert "matched_on" in block["arms"]["meso"]


def _write_twins(directory: Path, humans: Path, shift: dict[str, int], reps: int = 2) -> None:
    """Twin documents equal to the human's answers, moved by ``shift[pid]`` points."""
    directory.mkdir(parents=True)
    for path in sorted(humans.glob("pid_*.json")):
        pid = path.name.split("_")[1]
        document = json.loads(path.read_text())
        for rep in range(1, reps + 1):
            twin = copy.deepcopy(document)
            for ref in walk_questions(twin, include_descriptive=False):
                answers = ref.question["Answers"]
                answers["SelectedByPosition"] = [
                    min(7, v + shift[pid]) for v in answers["SelectedByPosition"]
                ]
            (directory / f"pid_{pid}_rep{rep}_response.json").write_text(json.dumps(twin))


class TestCompareArms:
    def test_contrasts_point_the_right_way(self, study: dict) -> None:
        humans = study["root"] / "split" / "holdout_json"
        runs = study["root"] / "twins"
        _write_twins(runs / "meso", humans, {p: 0 for p in PIDS})
        _write_twins(runs / "worse", humans, {p: 3 for p in PIDS})
        summary, table = compare_arms(
            human_dir=humans,
            arms={"meso": runs / "meso", "worse": runs / "worse"},
            contrasts={"H2": {"focal": "meso", "comparator": "worse"}},
        )
        meso = summary.set_index("arm").loc["meso"]
        assert meso["exact_match_rate"] == 1.0 and meso["mean_abs_deviation"] == 0.0
        assert meso["modal_rate"] == 1.0
        assert len(table) == len(PIDS)
        assert table["supports_agreement"].all() and table["supports_mad"].all()
        assert (table["exact_match_diff"] > 0).all() and (table["mad_diff"] < 0).all()

    def test_different_participants_refused(self, study: dict) -> None:
        humans = study["root"] / "split" / "holdout_json"
        runs = study["root"] / "twins"
        _write_twins(runs / "a", humans, {p: 0 for p in PIDS})
        _write_twins(runs / "b", humans, {p: 0 for p in PIDS})
        for f in (runs / "b").glob(f"pid_{PIDS[0]}_*"):
            f.unlink()
        with pytest.raises(ValueError, match="same participants"):
            compare_arms(
                human_dir=humans,
                arms={"a": runs / "a", "b": runs / "b"},
                contrasts={"H": {"focal": "a", "comparator": "b"}},
            )

    def test_unknown_arm_refused(self, study: dict) -> None:
        with pytest.raises(ValueError, match="not one of the arms"):
            compare_arms(
                human_dir=study["root"],
                arms={"a": study["root"]},
                contrasts={"H1": {"focal": "a", "comparator": "missing"}},
            )

    def test_end_to_end_all_three_hypotheses_in_stub_mode(self, study: dict) -> None:
        # Plumbing only: a stub fills every answer identically, so the
        # differences are zero. What is checked is that four real arms run and
        # the comparison accepts them as covering the same people and items.
        arms = {}
        for name, kwargs in {
            "meso": {"include": True},
            "macro": {"include": False},
            "no_persona": {"include": False, "empty": True},
            "scrambled": {"include": True, "scramble": True},
        }.items():
            config = _config(study, **kwargs)
            data = yaml.safe_load(config.read_text())
            data["text_simulation"]["LLM_simulation"]["replicates"] = 2
            config.write_text(yaml.safe_dump(data))
            run(config, stub=True)
            arms[name] = data["text_simulation"]["LLM_simulation"]["output_updated_questions_dir"]

        summary, table = compare_arms(
            human_dir=study["root"] / "split" / "holdout_json",
            arms=arms,
            contrasts={
                "H1": {"focal": "meso", "comparator": "macro"},
                "H2": {"focal": "meso", "comparator": "no_persona"},
                "H3": {"focal": "meso", "comparator": "scrambled"},
            },
        )
        assert list(summary["arm"]) == ["meso", "macro", "no_persona", "scrambled"]
        assert (summary["replicates"] == 2).all()
        assert sorted(table["contrast"].unique()) == ["H1", "H2", "H3"]
        assert len(table) == 3 * len(PIDS)
        assert (table["exact_match_diff"] == 0).all()
